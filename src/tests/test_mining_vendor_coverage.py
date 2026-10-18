"""Tests for mining.vendor_coverage -- the pre-scoring preflight that names, per
symbol, which scheduled NYSE sessions the vendor history lacks (and any bar on a
closed day, and an in-progress live bar), so a scoring run does not discover a
vendor hole only when a driver refuses.

Pure functions over synthetic indices; no network. The "one interior session
missing" shape mirrors a real vendor failure mode: a handful of ETFs and
indices lack one interior session while SPY has it.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mining import vendor_coverage as vc

WINDOW = ("2026-09-15", "2026-09-23")
# NYSE sessions 2026-09-15 .. 2026-09-23 (Tue..Wed of the next week): 7 sessions
FULL = ["2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18",
        "2026-09-21", "2026-09-22", "2026-09-23"]
ET = "America/New_York"


def idx(dates):
    return pd.DatetimeIndex(pd.to_datetime(dates))


def report(indices, as_of="2026-09-24 20:00", window=WINDOW):
    return vc.coverage_report(indices, start=window[0], end=window[1],
                              as_of=pd.Timestamp(as_of, tz=ET))


def test_complete_history_is_ok():
    out = report({"SPY": idx(FULL)})
    assert out["ok"] is True
    assert out["symbols"]["SPY"]["missing"] == []
    assert out["symbols"]["SPY"]["extra"] == []
    assert out["missing_by_date"] == {}


def test_an_interior_missing_session_is_named_per_symbol_and_per_date():
    without_922 = [d for d in FULL if d != "2026-09-22"]
    out = report({"SPY": idx(FULL), "IEF": idx(without_922), "AGG": idx(without_922)})
    assert out["ok"] is False
    assert out["symbols"]["SPY"]["missing"] == []
    assert out["symbols"]["IEF"]["missing"] == ["2026-09-22"]
    assert out["missing_by_date"] == {"2026-09-22": ["AGG", "IEF"]}


def test_a_bar_on_a_closed_day_is_reported_as_extra():
    labor_day = "2026-09-07"
    out = report({"^VIX": idx(["2026-09-04", labor_day, "2026-09-08"])},
                 window=("2026-09-04", "2026-09-08"))
    assert out["ok"] is False
    assert out["symbols"]["^VIX"]["extra"] == [labor_day]
    assert out["symbols"]["^VIX"]["missing"] == []


def test_only_the_requested_window_is_inspected():
    # a hole BEFORE the window must not be reported
    early_hole = [d for d in FULL]           # window data is complete
    out = report({"SPY": idx(["2026-09-01"] + early_hole)})
    assert out["ok"] is True


def test_a_symbol_with_no_rows_in_the_window_is_flagged_not_silently_ok():
    out = report({"NEW": idx(["2026-01-05"])})
    assert out["ok"] is False
    assert out["symbols"]["NEW"]["missing"] == [
        "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18",
        "2026-09-21", "2026-09-22", "2026-09-23"]


def test_timezone_aware_and_intraday_stamped_indices_are_normalised():
    stamped = pd.DatetimeIndex(pd.to_datetime([d + " 00:00:00" for d in FULL])
                               ).tz_localize(ET)
    assert report({"SPY": stamped})["ok"] is True


def test_todays_bar_before_the_close_is_flagged_as_a_live_partial_bar():
    today = ["2026-09-23"]
    early = report({"SPY": idx(FULL)}, as_of="2026-09-23 11:00")
    assert early["symbols"]["SPY"]["live_partial_bar"] is True
    assert early["ok"] is False


def test_todays_bar_after_the_close_is_final():
    late = report({"SPY": idx(FULL)}, as_of="2026-09-23 16:30")
    assert late["symbols"]["SPY"]["live_partial_bar"] is False
    assert late["ok"] is True


def test_a_session_that_has_not_settled_yet_is_not_reported_missing():
    # window ends TODAY (2026-09-24, a Thursday session) and it is 09:00 ET:
    # the day has not traded, so the absent bar is not a vendor hole
    out = report({"SPY": idx(FULL)}, as_of="2026-09-24 09:00",
                 window=("2026-09-15", "2026-09-24"))
    assert out["symbols"]["SPY"]["missing"] == []
    assert out["ok"] is True


def test_a_session_still_open_is_not_reported_missing_either():
    out = report({"SPY": idx(FULL)}, as_of="2026-09-24 11:00",
                 window=("2026-09-15", "2026-09-24"))
    assert out["symbols"]["SPY"]["missing"] == []


def test_an_unsettled_days_live_bar_is_live_not_a_bar_on_a_closed_day():
    # ^VIX-style: a bar dated today exists while the session is still open
    out = report({"^VIX": idx(FULL + ["2026-09-24"])}, as_of="2026-09-24 11:00",
                 window=("2026-09-15", "2026-09-24"))
    row = out["symbols"]["^VIX"]
    assert row["extra"] == []
    assert row["live_partial_bar"] is True


def test_a_bar_on_a_future_session_date_is_not_extra_either():
    out = report({"SPY": idx(FULL + ["2026-09-25"])}, as_of="2026-09-24 11:00",
                 window=("2026-09-15", "2026-09-29"))
    assert out["symbols"]["SPY"]["extra"] == []


def test_a_settled_session_with_no_bar_is_missing():
    out = report({"SPY": idx(FULL)}, as_of="2026-09-24 17:00",
                 window=("2026-09-15", "2026-09-24"))
    assert out["symbols"]["SPY"]["missing"] == ["2026-09-24"]
    assert out["ok"] is False


def test_future_sessions_are_never_missing():
    out = report({"SPY": idx(FULL)}, as_of="2026-09-24 17:00",
                 window=("2026-09-15", "2026-09-29"))
    assert out["symbols"]["SPY"]["missing"] == ["2026-09-24"]   # not 25, 28, 29


def test_a_prior_days_last_bar_is_never_live():
    out = report({"SPY": idx(FULL)}, as_of="2026-09-24 09:00")
    assert out["symbols"]["SPY"]["live_partial_bar"] is False


def test_the_report_carries_first_and_last_dates_for_the_reader():
    out = report({"SPY": idx(FULL)})
    assert out["symbols"]["SPY"]["last"] == "2026-09-23"
    assert out["symbols"]["SPY"]["n_in_window"] == 7


def test_an_empty_symbol_map_is_refused():
    with pytest.raises(ValueError):
        report({})


def test_window_end_before_start_is_refused():
    with pytest.raises(ValueError):
        vc.coverage_report({"SPY": idx(FULL)}, start="2026-09-23",
                           end="2026-09-15", as_of=pd.Timestamp("2026-09-24", tz=ET))


def _real_fetch_ohlc_shape(dates):
    """The shape data_pipeline.fetch_ohlc REALLY returns: python dates in a
    ``Date`` COLUMN over a plain integer index (not a DatetimeIndex). Reading
    that integer index as dates would yield 1970 epoch dates."""
    import datetime
    stamps = [datetime.date.fromisoformat(d) for d in dates]
    return pd.DataFrame({"Date": stamps, "Open": 1.0, "High": 1.0,
                         "Low": 1.0, "Close": 1.0})


def test_index_from_frame_prefers_the_date_column_of_the_real_fetch_shape():
    frame = _real_fetch_ohlc_shape(FULL)
    got = vc.index_from_frame(frame)
    assert list(got.strftime("%Y-%m-%d")) == FULL


def test_index_from_frame_falls_back_to_a_datetime_index():
    frame = pd.DataFrame({"Close": 1.0}, index=idx(FULL))
    assert list(vc.index_from_frame(frame).strftime("%Y-%m-%d")) == FULL


def test_index_from_frame_refuses_an_integer_index_with_no_date_column():
    with pytest.raises(ValueError):
        vc.index_from_frame(pd.DataFrame({"Close": [1.0, 2.0, 3.0]}))


def _load_script():
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts",
                        "check_vendor_coverage.py")
    spec = importlib.util.spec_from_file_location("check_vendor_coverage", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_main_reads_the_real_fetch_shape_end_to_end(capsys):
    script = _load_script()
    without_922 = [d for d in FULL if d != "2026-09-22"]
    frames = {"SPY": _real_fetch_ohlc_shape(FULL),
              "IEF": _real_fetch_ohlc_shape(without_922)}
    args = ["--symbols", "SPY,IEF", "--start", WINDOW[0], "--end", WINDOW[1],
            "--as-of", "2026-09-24 20:00"]
    assert script.main(args, fetch=lambda s: frames[s]) == 1
    out = capsys.readouterr().out
    assert "no bar on 2026-09-22: IEF" in out
    assert "last=2026-09-23" in out
    assert "1970" not in out


def test_main_exit_codes_use_an_injected_fetcher(tmp_path, capsys):
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts",
                        "check_vendor_coverage.py")
    spec = importlib.util.spec_from_file_location("check_vendor_coverage", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    without_922 = [d for d in FULL if d != "2026-09-22"]
    frames = {"SPY": idx(FULL), "IEF": idx(without_922)}

    def fetch(symbol):
        return pd.DataFrame({"Close": 1.0}, index=frames[symbol])

    args_bad = ["--symbols", "SPY,IEF", "--start", WINDOW[0], "--end", WINDOW[1],
                "--as-of", "2026-09-24 20:00"]
    assert script.main(args_bad, fetch=fetch) == 1
    assert "2026-09-22" in capsys.readouterr().out
    args_ok = ["--symbols", "SPY", "--start", WINDOW[0], "--end", WINDOW[1],
               "--as-of", "2026-09-24 20:00"]
    assert script.main(args_ok, fetch=fetch) == 0


# ---- last_complete_session / --suggest-cutoff --------------------------------
# The cutoff is only ever a SUGGESTION for the opt-in scoring setting: the last
# actual session before the first hole, never a repaired or filled date.

def _full_minus(*missing):
    return [d for d in FULL if d not in missing]


def test_last_complete_session_missing_interior_bar_shape_is_the_session_before_the_hole():
    # a realistic vendor-hole shape: two symbols lack one interior session
    # while SPY has it and later bars are present
    out = report({"SPY": idx(FULL), "IEF": idx(_full_minus("2026-09-22")),
                  "AGG": idx(_full_minus("2026-09-22"))})
    assert out["ok"] is False
    assert vc.last_complete_session(out) == "2026-09-21"


def test_last_complete_session_ignores_later_bars_after_the_hole():
    later = FULL + ["2026-09-24"]
    out = report({"SPY": idx(later),
                  "IEF": idx([d for d in later if d != "2026-09-22"])},
                 as_of="2026-09-24 20:00", window=("2026-09-15", "2026-09-24"))
    assert vc.last_complete_session(out) == "2026-09-21"


def test_last_complete_session_with_no_holes_is_the_last_settled_session():
    out = report({"SPY": idx(FULL), "IEF": idx(FULL)})
    assert out["ok"] is True
    assert vc.last_complete_session(out) == "2026-09-23"


def test_last_complete_session_no_holes_stops_at_the_last_settled_session():
    # window runs past today's still-unsettled session and into the future
    out = report({"SPY": idx(FULL)}, as_of="2026-09-24 09:00",
                 window=("2026-09-15", "2026-09-29"))
    assert out["ok"] is True
    assert vc.last_complete_session(out) == "2026-09-23"


def test_last_complete_session_does_not_count_a_live_partial_bar():
    # 2026-09-23 has a bar but the ET clock has not reached its close: no
    # missing session, yet the session is not complete
    out = report({"SPY": idx(FULL)}, as_of="2026-09-23 11:00")
    assert out["symbols"]["SPY"]["live_partial_bar"] is True
    assert out["symbols"]["SPY"]["missing"] == []
    assert vc.last_complete_session(out) == "2026-09-22"


def test_last_complete_session_live_bar_on_a_later_day_than_the_window_data():
    out = report({"^VIX": idx(FULL + ["2026-09-24"]), "SPY": idx(FULL)},
                 as_of="2026-09-24 11:00", window=("2026-09-15", "2026-09-24"))
    assert out["symbols"]["^VIX"]["live_partial_bar"] is True
    assert vc.last_complete_session(out) == "2026-09-23"


def test_last_complete_session_is_none_when_the_first_session_is_a_hole():
    # window starts on a session (2026-09-15) that IEF lacks: nothing before it
    out = report({"SPY": idx(FULL), "IEF": idx(_full_minus("2026-09-15"))})
    assert vc.last_complete_session(out) is None


def test_last_complete_session_is_none_for_a_bar_on_a_closed_window_start_day():
    labor_day = "2026-09-07"
    out = report({"^VIX": idx([labor_day, "2026-09-08"])},
                 window=(labor_day, "2026-09-08"))
    assert out["symbols"]["^VIX"]["extra"] == [labor_day]
    assert vc.last_complete_session(out) is None


def test_last_complete_session_hole_on_the_second_session_leaves_the_first():
    out = report({"IEF": idx(_full_minus("2026-09-16"))})
    assert vc.last_complete_session(out) == "2026-09-15"


def test_last_complete_session_takes_the_earliest_hole_across_symbols():
    out = report({"SPY": idx(FULL),
                  "IEF": idx(_full_minus("2026-09-22")),
                  "AGG": idx(_full_minus("2026-09-17"))})
    assert vc.last_complete_session(out) == "2026-09-16"


def test_last_complete_session_a_mid_window_bar_on_a_closed_day_is_a_hole():
    labor_day = "2026-09-07"
    out = report({"^VIX": idx(["2026-09-02", "2026-09-03", "2026-09-04",
                               labor_day, "2026-09-08", "2026-09-09"])},
                 window=("2026-09-02", "2026-09-09"))
    assert out["symbols"]["^VIX"]["extra"] == [labor_day]
    assert vc.last_complete_session(out) == "2026-09-04"


def test_last_complete_session_is_none_when_nothing_in_the_window_has_settled():
    out = report({"SPY": idx(FULL)}, as_of="2026-09-24 09:00",
                 window=("2026-09-25", "2026-09-29"))
    assert vc.last_complete_session(out) is None


def test_last_complete_session_needs_the_report_coverage_report_returns():
    with pytest.raises(ValueError):
        vc.last_complete_session({"window": ["2026-09-15", "2026-09-23"],
                                  "symbols": {}})


def test_coverage_report_states_what_it_treated_as_settled():
    assert report({"SPY": idx(FULL)}, as_of="2026-09-24 20:00")[
        "settled_through"] == "2026-09-24"
    assert report({"SPY": idx(FULL)}, as_of="2026-09-24 09:00")[
        "settled_through"] == "2026-09-23"


def _frames_fetch(frames):
    def fetch(symbol):
        if isinstance(frames[symbol], Exception):
            raise frames[symbol]
        return _real_fetch_ohlc_shape(frames[symbol])
    return fetch


def _cli_args(symbols, *extra):
    return ["--symbols", symbols, "--start", WINDOW[0], "--end", WINDOW[1],
            "--as-of", "2026-09-24 20:00", *extra]


def test_suggest_cutoff_prints_one_line_and_keeps_the_hole_exit_code(capsys):
    script = _load_script()
    fetch = _frames_fetch({"SPY": FULL, "IEF": _full_minus("2026-09-22")})
    assert script.main(_cli_args("SPY,IEF", "--suggest-cutoff"), fetch=fetch) == 1
    out = capsys.readouterr().out
    assert out.count("last complete session:") == 1
    assert out.rstrip("\n").splitlines()[-1] == "last complete session: 2026-09-21"


def test_suggest_cutoff_on_clean_history_exits_zero(capsys):
    script = _load_script()
    fetch = _frames_fetch({"SPY": FULL})
    assert script.main(_cli_args("SPY", "--suggest-cutoff"), fetch=fetch) == 0
    assert capsys.readouterr().out.rstrip("\n").splitlines()[-1] == (
        "last complete session: 2026-09-23")


def test_suggest_cutoff_none_in_window_is_stated_clearly(capsys):
    script = _load_script()
    fetch = _frames_fetch({"SPY": FULL, "IEF": _full_minus("2026-09-15")})
    assert script.main(_cli_args("SPY,IEF", "--suggest-cutoff"), fetch=fetch) == 1
    assert capsys.readouterr().out.rstrip("\n").splitlines()[-1] == (
        "last complete session: none in window")


def test_output_without_the_flag_has_no_cutoff_line_and_the_flag_only_appends(capsys):
    script = _load_script()
    fetch = _frames_fetch({"SPY": FULL, "IEF": _full_minus("2026-09-22")})
    assert script.main(_cli_args("SPY,IEF"), fetch=fetch) == 1
    plain = capsys.readouterr().out
    assert "last complete session" not in plain
    assert script.main(_cli_args("SPY,IEF", "--suggest-cutoff"), fetch=fetch) == 1
    flagged = capsys.readouterr().out
    assert flagged == plain + "last complete session: 2026-09-21\n"


def test_suggest_cutoff_never_hides_a_fetch_failure(capsys):
    script = _load_script()
    fetch = _frames_fetch({"SPY": FULL, "IEF": RuntimeError("vendor down")})
    assert script.main(_cli_args("SPY,IEF", "--suggest-cutoff"), fetch=fetch) == 2
    out = capsys.readouterr().out
    assert "FETCH FAILED IEF" in out
    last = out.rstrip("\n").splitlines()[-1]
    # a partial symbol set cannot vouch for a complete session
    assert last.startswith("last complete session: unavailable")
    assert "2026-09-2" not in last


def test_suggest_cutoff_with_no_fetchable_symbol_still_exits_two(capsys):
    script = _load_script()
    fetch = _frames_fetch({"SPY": RuntimeError("vendor down")})
    assert script.main(_cli_args("SPY", "--suggest-cutoff"), fetch=fetch) == 2
    assert "last complete session: unavailable" in capsys.readouterr().out


# ---- a bar is counted the way the scoring drivers count it -------------------
# The drivers (run_batch5_mining.load_frame) DROP every row whose OHLC fields are
# all blank before they read a single date, so a session the vendor listed with
# no prices is a session the driver refuses. The preflight must not call it
# present, or it would suggest a cutoff the driver then rejects.

NAN = float("nan")
OHLC = ["Open", "High", "Low", "Close"]


def _fetch_shape_with_blank_rows(dates, blank=(), partial=()):
    """The real fetch shape (python dates in a ``Date`` column over an integer
    index) where the rows dated in ``blank`` carry NO prices at all (a session
    the vendor never printed) and the rows in ``partial`` carry a Close but no
    Open (a bad print, which the driver keeps and later refuses)."""
    frame = _real_fetch_ohlc_shape(dates)
    stamps = frame["Date"].map(lambda d: f"{d:%Y-%m-%d}")
    frame.loc[stamps.isin(blank), OHLC] = NAN
    frame.loc[stamps.isin(partial), "Open"] = NAN
    return frame


def _driver_module():
    scripts = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                           "scripts"))
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import run_batch5_mining
    return run_batch5_mining


def _cli_args_at(symbols, as_of, window, *extra):
    return ["--symbols", symbols, "--start", window[0], "--end", window[1],
            "--as-of", as_of, *extra]


def _parity_frames():
    with_blank = _fetch_shape_with_blank_rows(FULL, blank=("2026-09-22",))
    as_datetime_index = with_blank.set_index(pd.to_datetime(with_blank["Date"])
                                             ).drop(columns="Date")
    return {
        "date column": with_blank,
        "datetime index": as_datetime_index,
        "partial row": _fetch_shape_with_blank_rows(FULL, partial=("2026-09-22",)),
        "two blank rows": _fetch_shape_with_blank_rows(
            FULL, blank=("2026-09-16", "2026-09-22")),
        "no blank rows": _real_fetch_ohlc_shape(FULL),
        "close only": pd.DataFrame({"Close": [1.0, NAN, 1.0]}, index=idx(FULL[:3])),
        "no ohlc columns": pd.DataFrame({"Volume": [1.0, 2.0]}, index=idx(FULL[:2])),
        "empty": pd.DataFrame({"Open": [], "Close": []}),
    }


def test_drop_all_blank_rows_matches_the_drivers_own_helper_exactly():
    driver = _driver_module()
    for name, frame in _parity_frames().items():
        kept, dropped = vc.drop_all_blank_rows(frame)
        want_kept, want_dropped = driver._drop_all_nan_rows(frame)
        pd.testing.assert_frame_equal(kept, want_kept, obj=name)
        assert dropped == want_dropped, name
    assert vc.drop_all_blank_rows(None) == driver._drop_all_nan_rows(None)


def test_drop_all_blank_rows_names_the_dropped_dates_and_never_edits_its_input():
    frame = _fetch_shape_with_blank_rows(FULL, blank=("2026-09-16", "2026-09-22"))
    before = frame.copy()
    kept, dropped = vc.drop_all_blank_rows(frame)
    assert dropped == ["2026-09-16", "2026-09-22"]
    assert len(kept) == len(FULL) - 2
    pd.testing.assert_frame_equal(frame, before)


def test_driver_index_excludes_a_session_the_vendor_listed_with_no_prices():
    frame = _fetch_shape_with_blank_rows(FULL, blank=("2026-09-22",))
    assert "2026-09-22" in list(vc.index_from_frame(frame).strftime("%Y-%m-%d"))
    got = vc.driver_index_from_frame(frame)
    assert list(got.strftime("%Y-%m-%d")) == _full_minus("2026-09-22")


def test_driver_index_keeps_a_partly_blank_row_exactly_as_the_driver_does():
    # a bad print is kept by the driver's blank-row drop and refused later, so
    # the preflight must not silently hide it either
    frame = _fetch_shape_with_blank_rows(FULL, partial=("2026-09-22",))
    assert list(vc.driver_index_from_frame(frame).strftime("%Y-%m-%d")) == FULL


def test_driver_index_reads_a_datetime_indexed_frame_too():
    frame = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0},
                         index=idx(FULL))
    frame.loc[pd.Timestamp("2026-09-22"), OHLC] = NAN
    assert list(vc.driver_index_from_frame(frame).strftime("%Y-%m-%d")) == (
        _full_minus("2026-09-22"))


def test_driver_index_refuses_a_history_with_no_priced_row_at_all():
    frame = _fetch_shape_with_blank_rows(FULL, blank=tuple(FULL))
    with pytest.raises(ValueError, match="no row with any price"):
        vc.driver_index_from_frame(frame)


def test_an_all_blank_0922_row_is_a_missing_session_and_the_cutoff_is_0921(capsys):
    script = _load_script()
    frames = {"SPY": _real_fetch_ohlc_shape(FULL),
              "IEF": _fetch_shape_with_blank_rows(FULL, blank=("2026-09-22",))}
    args = _cli_args("SPY,IEF", "--suggest-cutoff")
    assert script.main(args, fetch=lambda s: frames[s]) == 1
    out = capsys.readouterr().out
    assert "no bar on 2026-09-22: IEF" in out
    assert "MISSING 2026-09-22" in out
    assert out.rstrip("\n").splitlines()[-1] == "last complete session: 2026-09-21"


def test_a_clean_history_with_a_partly_blank_row_still_reads_as_present(capsys):
    script = _load_script()
    frames = {"SPY": _fetch_shape_with_blank_rows(FULL, partial=("2026-09-22",))}
    args = _cli_args("SPY", "--suggest-cutoff")
    assert script.main(args, fetch=lambda s: frames[s]) == 0
    assert capsys.readouterr().out.rstrip("\n").splitlines()[-1] == (
        "last complete session: 2026-09-23")


def test_a_history_of_only_blank_rows_is_a_failed_fetch_not_a_hole_report(capsys):
    script = _load_script()
    frames = {"SPY": _real_fetch_ohlc_shape(FULL),
              "IEF": _fetch_shape_with_blank_rows(FULL, blank=tuple(FULL))}
    assert script.main(_cli_args("SPY,IEF", "--suggest-cutoff"),
                       fetch=lambda s: frames[s]) == 2
    out = capsys.readouterr().out
    assert "FETCH FAILED IEF" in out
    assert out.rstrip("\n").splitlines()[-1].startswith(
        "last complete session: unavailable")


def test_a_bill_series_without_a_bond_market_holiday_is_missing_and_errs_safe(capsys):
    # 2026-10-12 (Columbus Day): the NYSE is open, the bond market is closed, and
    # a bill-rate series such as ^IRX has no bar. The treasury driver tolerates
    # that only with no history end set and refuses a cutoff on such a day, so
    # the preflight reports it missing and suggests the session before it.
    script = _load_script()
    window = ("2026-10-08", "2026-10-14")
    sessions = ["2026-10-08", "2026-10-09", "2026-10-12", "2026-10-13", "2026-10-14"]
    frames = {"SPY": _real_fetch_ohlc_shape(sessions),
              "^IRX": _real_fetch_ohlc_shape([d for d in sessions
                                              if d != "2026-10-12"])}
    args = _cli_args_at("SPY,^IRX", "2026-10-15 20:00", window, "--suggest-cutoff")
    assert script.main(args, fetch=lambda s: frames[s]) == 1
    out = capsys.readouterr().out
    assert "no bar on 2026-10-12: ^IRX" in out
    assert out.rstrip("\n").splitlines()[-1] == "last complete session: 2026-10-09"


def test_the_script_docstring_states_the_blank_row_and_bond_holiday_rules():
    doc = " ".join(_load_script().__doc__.split())
    assert "no prices at all" in doc
    assert "bond-market holiday" in doc and "^IRX" in doc
    assert "safe side" in doc
