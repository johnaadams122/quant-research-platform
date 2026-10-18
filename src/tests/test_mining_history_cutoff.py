"""Tests for mining.history_cutoff -- the opt-in "end the scored history at a
chosen last session" setting.

Why it exists: a data vendor can lack one interior bar (in these tests,
2026-09-22) for many series while having the bars after it, and the scoring
drivers (correctly) refuse a calendar hole. The setting only DROPS sessions dated after the cutoff -- never
fills, patches, shifts or repairs -- and it fails closed: the cutoff must be a
real NYSE session and every series must have a bar exactly on it.

Pure functions over synthetic frames; no network, no file I/O. The
missing-interior-bar shape is exercised: one series with the interior bar, one without,
both carrying later bars, cutoff 2026-09-21.
"""
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mining import history_cutoff as hc
from mining import vendor_coverage as vc

CUTOFF = "2026-09-21"
# NYSE sessions 2026-09-15 .. 2026-09-24
SESSIONS = ["2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18",
            "2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24"]
WITHOUT_0922 = [d for d in SESSIONS if d != "2026-09-22"]


def make_frame(dates, shape="index"):
    """Synthetic OHLC frame with distinct, deterministic values per row.
    ``shape="index"`` -> DatetimeIndex (what the drivers hold after
    ``_normalize_ohlc``); ``shape="column"`` -> a ``Date`` column of python
    dates over a RangeIndex (what ``data_pipeline.fetch_ohlc`` returns)."""
    n = len(dates)
    base = np.arange(n, dtype=float) + 100.0
    values = {"Open": base, "High": base + 1.5, "Low": base - 1.5,
              "Close": base + 0.5}
    stamps = pd.to_datetime(list(dates))
    if shape == "index":
        return pd.DataFrame(values, index=pd.DatetimeIndex(stamps))
    frame = pd.DataFrame(values)
    frame.insert(0, "Date", [s.date() for s in stamps])
    return frame


def ts(text):
    return pd.Timestamp(text)


# ------------------------------------------------------------------- parsing

def test_none_is_the_only_off_value():
    assert hc.parse_history_end(None) is None


def test_valid_string_parses_to_a_normalized_naive_timestamp():
    out = hc.parse_history_end(CUTOFF)
    assert out == ts(CUTOFF)
    assert isinstance(out, pd.Timestamp)
    assert out.tzinfo is None
    assert out == out.normalize()


@pytest.mark.parametrize("value", [
    datetime.date(2026, 9, 21),
    datetime.datetime(2026, 9, 21),
    pd.Timestamp("2026-09-21"),
])
def test_date_datetime_and_timestamp_at_midnight_are_accepted(value):
    out = hc.parse_history_end(value)
    assert out == ts(CUTOFF)
    assert out.tzinfo is None


# the compact eight-digit form (no separators); only YYYY-MM-DD is accepted
COMPACT = "2026" + "0921"

INVALID_INPUTS = [
    "", " ", "  \t", COMPACT, "2026-9-21", "2026-09-2", "26-09-21",
    "2026/09/21", "2026.09.21", "21-09-2026", "09-21-2026",
    " 2026-09-21", "2026-09-21 ", "2026-09-21\n", "\n2026-09-21",
    "2026-09-21T00:00:00", "2026-09-21 00:00", "2026-09-21 16:00",
    "2026-09-21Z", "2026-13-01", "2026-00-10", "2026-02-30", "2026-09-31",
    "2027-02-29", "2026-09-00", "abc", "None", "null", "today",
    "2026-09-21,2026-09-22", "2026-W38-1",
    "٢٠٢٦-٠٩-٢١",  # non-ASCII digits
    2026, 1.5, True, False, b"2026-09-21", ["2026-09-21"], ("2026-09-21",),
    {"date": "2026-09-21"},
    pd.Timestamp("2026-09-21 16:00"),
    pd.Timestamp("2026-09-21 00:00:01"),
    pd.Timestamp("2026-09-21", tz="America/New_York"),
    datetime.datetime(2026, 9, 21, 9, 30),
    pd.NaT,
    np.datetime64("2026-09-21"),
]


@pytest.mark.parametrize("value", INVALID_INPUTS,
                         ids=[f"form{i:02d}" for i in range(len(INVALID_INPUTS))])
def test_every_invalid_form_is_refused_as_history_end_invalid(value):
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.parse_history_end(value)
    assert info.value.reason_code == "history_end_invalid"
    assert info.value.symbols == ()


@pytest.mark.parametrize("value,label", [
    ("2026-09-19", "Saturday"),
    ("2026-09-20", "Sunday"),
    ("2026-09-07", "Labor Day"),
    ("2026-12-25", "Christmas"),
    ("2012-10-29", "Hurricane Sandy unscheduled closure"),
    ("1700-01-03", "before the calendar begins"),
    ("0001-01-01", "real date, not representable as a timestamp"),
    ("9999-12-31", "real date, not representable as a timestamp"),
])
def test_a_real_date_that_is_not_a_session_is_refused(value, label):
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.parse_history_end(value)
    assert info.value.reason_code == "history_end_not_a_session", label
    assert info.value.symbols == ()


def test_a_date_object_on_a_holiday_is_also_refused():
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.parse_history_end(datetime.date(2026, 9, 7))
    assert info.value.reason_code == "history_end_not_a_session"


def test_error_type_is_a_value_error_with_the_documented_attributes():
    assert issubclass(hc.HistoryCutoffError, ValueError)
    err = hc.HistoryCutoffError("history_end_invalid", "msg", symbols=["B", "A"])
    assert err.reason_code == "history_end_invalid"
    assert err.symbols == ("B", "A")
    assert str(err) == "msg"
    assert hc.HistoryCutoffError("history_end_empty", "m").symbols == ()


# --------------------------------------------------------------- trim: one frame

@pytest.mark.parametrize("shape", ["index", "column"])
def test_trim_is_a_pure_drop_of_the_exact_prefix(shape):
    frame = make_frame(SESSIONS, shape)
    trimmed, record = hc.trim_frame_to_history_end(frame, ts(CUTOFF), symbol="SPY")
    kept = SESSIONS.index(CUTOFF) + 1
    pd.testing.assert_frame_equal(trimmed, frame.iloc[:kept])
    assert list(trimmed.dtypes) == list(frame.dtypes)
    assert trimmed.index.equals(frame.index[:kept])
    assert len(trimmed) == kept
    assert record["dropped_sessions"] == len(SESSIONS) - kept


@pytest.mark.parametrize("shape", ["index", "column"])
def test_no_row_dated_after_the_cutoff_survives(shape):
    frame = make_frame(SESSIONS, shape)
    trimmed, _ = hc.trim_frame_to_history_end(frame, ts(CUTOFF))
    dates = vc.index_from_frame(trimmed)
    assert dates.max() == ts(CUTOFF)
    assert not (dates > ts(CUTOFF)).any()


@pytest.mark.parametrize("shape", ["index", "column"])
def test_input_frame_is_never_mutated_and_the_result_is_a_copy(shape):
    frame = make_frame(SESSIONS, shape)
    before = frame.copy(deep=True)
    trimmed, _ = hc.trim_frame_to_history_end(frame, ts(CUTOFF))
    pd.testing.assert_frame_equal(frame, before)
    assert trimmed is not frame
    trimmed.iloc[0, trimmed.columns.get_loc("Close")] = -1.0
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("shape", ["index", "column"])
def test_cutoff_equal_to_the_last_bar_is_a_valid_no_op(shape):
    frame = make_frame(SESSIONS[:SESSIONS.index(CUTOFF) + 1], shape)
    before = frame.copy(deep=True)
    trimmed, record = hc.trim_frame_to_history_end(frame, ts(CUTOFF), symbol="SPY")
    pd.testing.assert_frame_equal(trimmed, before)
    assert trimmed is not frame
    assert record["dropped_sessions"] == 0
    assert record["dropped_dates"] == []
    assert record["raw_last"] == CUTOFF
    assert record["kept_last"] == CUTOFF


@pytest.mark.parametrize("shape", ["index", "column"])
def test_record_carries_iso_strings_and_the_dropped_dates(shape):
    frame = make_frame(SESSIONS, shape)
    _, record = hc.trim_frame_to_history_end(frame, ts(CUTOFF))
    assert record == {
        "raw_first": "2026-09-15",
        "raw_last": "2026-09-24",
        "kept_last": CUTOFF,
        "dropped_sessions": 3,
        "dropped_dates": ["2026-09-22", "2026-09-23", "2026-09-24"],
    }
    assert isinstance(record["dropped_sessions"], int)
    json.dumps(record)


@pytest.mark.parametrize("shape", ["index", "column"])
def test_a_frame_missing_the_cutoff_date_is_refused_even_with_later_bars(shape):
    frame = make_frame([d for d in SESSIONS if d != CUTOFF], shape)
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.trim_frame_to_history_end(frame, ts(CUTOFF), symbol="IEF")
    assert info.value.reason_code == "history_end_not_covered"
    assert info.value.symbols == ("IEF",)
    assert "IEF" in str(info.value)


@pytest.mark.parametrize("shape", ["index", "column"])
def test_a_frame_ending_before_the_cutoff_is_refused(shape):
    frame = make_frame(["2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18"], shape)
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.trim_frame_to_history_end(frame, ts(CUTOFF), symbol="TLT")
    assert info.value.reason_code == "history_end_not_covered"
    assert info.value.symbols == ("TLT",)


def test_the_symbol_defaults_to_empty_in_a_refusal():
    frame = make_frame(["2026-09-15", "2026-09-16"])
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.trim_frame_to_history_end(frame, ts(CUTOFF))
    assert info.value.symbols == ("",)


@pytest.mark.parametrize("shape", ["index", "column"])
def test_an_empty_frame_is_refused_as_history_end_empty(shape):
    frame = make_frame(SESSIONS, shape).iloc[0:0]
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.trim_frame_to_history_end(frame, ts(CUTOFF), symbol="SPY")
    assert info.value.reason_code == "history_end_empty"
    assert info.value.symbols == ("SPY",)


def test_a_tz_aware_datetime_index_is_read_by_calendar_date():
    frame = make_frame(SESSIONS)
    frame.index = frame.index.tz_localize("America/New_York")
    trimmed, record = hc.trim_frame_to_history_end(frame, ts(CUTOFF))
    assert len(trimmed) == SESSIONS.index(CUTOFF) + 1
    assert trimmed.index.tz is not None
    assert record["kept_last"] == CUTOFF


def test_every_bar_of_a_kept_date_survives_when_dates_carry_times():
    stamps = pd.DatetimeIndex(["2026-09-18 09:30", "2026-09-18 15:59",
                               "2026-09-21 09:30", "2026-09-21 15:59",
                               "2026-09-22 09:30"])
    frame = pd.DataFrame({"Close": np.arange(5.0)}, index=stamps)
    trimmed, record = hc.trim_frame_to_history_end(frame, ts(CUTOFF))
    assert list(trimmed["Close"]) == [0.0, 1.0, 2.0, 3.0]
    assert record["dropped_dates"] == ["2026-09-22"]


@pytest.mark.parametrize("bad", [
    CUTOFF, datetime.date(2026, 9, 21), pd.Timestamp("2026-09-21 16:00"),
    pd.Timestamp("2026-09-21", tz="UTC"), pd.NaT, None, 2026,
], ids=lambda v: repr(v)[:30])
def test_an_unparsed_cutoff_is_refused_by_trim_and_provenance(bad):
    # the trim / provenance functions take what parse_history_end returns; a raw
    # string or date would skip the strict-format and NYSE-session checks
    frame = make_frame(SESSIONS)
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.trim_frame_to_history_end(frame, bad)
    assert info.value.reason_code == "history_end_invalid"
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.build_provenance(bad, _records())
    assert info.value.reason_code == "history_end_invalid"


def test_a_frame_with_no_readable_dates_is_a_plain_value_error():
    frame = pd.DataFrame({"Close": [1.0, 2.0]})          # integer index, no Date
    with pytest.raises(ValueError) as info:
        hc.trim_frame_to_history_end(frame, ts(CUTOFF))
    assert not isinstance(info.value, hc.HistoryCutoffError)
    with pytest.raises(ValueError):
        hc.trim_frame_to_history_end("not a frame", ts(CUTOFF))


def test_unparseable_bar_dates_are_refused_rather_than_silently_dropped():
    frame = make_frame(SESSIONS, "column")
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame.loc[2, "Date"] = pd.NaT
    with pytest.raises(ValueError, match="unreadable"):
        hc.trim_frame_to_history_end(frame, ts(CUTOFF))


# -------------------------------------------------------------- trim: many frames

def test_none_returns_the_identical_frame_objects_and_no_provenance():
    frames = {"SPY": make_frame(SESSIONS), "IEF": make_frame(WITHOUT_0922, "column")}
    out, provenance = hc.trim_frames_to_history_end(frames, None)
    assert provenance is None
    assert out is not frames
    assert list(out) == list(frames)
    for symbol in frames:
        assert out[symbol] is frames[symbol]
    out["NEW"] = make_frame(SESSIONS)                   # the returned dict is new
    assert "NEW" not in frames


def test_none_does_not_even_look_at_the_frames():
    # default off: a frame that would be refused under any cutoff passes through
    frames = {"BROKEN": pd.DataFrame({"Close": [1.0]})}
    out, provenance = hc.trim_frames_to_history_end(frames, None)
    assert out["BROKEN"] is frames["BROKEN"]
    assert provenance is None


def test_an_invalid_cutoff_is_refused_before_any_frame_is_read():
    frames = {"SPY": make_frame(SESSIONS)}
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.trim_frames_to_history_end(frames, "2026-9-21")
    assert info.value.reason_code == "history_end_invalid"
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.trim_frames_to_history_end(frames, "2026-09-19")
    assert info.value.reason_code == "history_end_not_a_session"
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.trim_frames_to_history_end(frames, "")           # empty is invalid, not off
    assert info.value.reason_code == "history_end_invalid"


def test_the_missing_interior_bar_shape_trims_every_series_to_the_cutoff():
    frames = {
        "SPY": make_frame(SESSIONS),               # has the 2026-09-22 bar
        "IEF": make_frame(WITHOUT_0922, "column"),  # lacks it, has 09-23/24
        "^IRX": make_frame(WITHOUT_0922),
    }
    before = {k: v.copy(deep=True) for k, v in frames.items()}
    out, provenance = hc.trim_frames_to_history_end(frames, CUTOFF)
    for symbol in frames:                              # inputs untouched
        pd.testing.assert_frame_equal(frames[symbol], before[symbol])
    for symbol, frame in out.items():
        assert vc.index_from_frame(frame).max() == ts(CUTOFF)
    # the interior hole disappears with the later bars: every trimmed series now
    # has the same calendar and passes the coverage preflight through the cutoff
    spy_dates = vc.index_from_frame(out["SPY"])
    assert vc.index_from_frame(out["IEF"]).equals(spy_dates)
    report = vc.coverage_report(
        {s: vc.index_from_frame(f) for s, f in out.items()},
        start="2026-09-15", end=CUTOFF, as_of=pd.Timestamp("2026-09-25 20:00",
                                                          tz="America/New_York"))
    assert report["ok"] is True
    assert provenance["history_end"] == CUTOFF
    assert provenance["by_symbol"]["SPY"]["dropped_dates"] == [
        "2026-09-22", "2026-09-23", "2026-09-24"]
    assert provenance["by_symbol"]["IEF"]["dropped_dates"] == [
        "2026-09-23", "2026-09-24"]
    assert provenance["by_symbol"]["IEF"]["raw_last"] == "2026-09-24"
    assert provenance["dropped_dates"] == ["2026-09-22", "2026-09-23", "2026-09-24"]
    assert provenance["total_dropped_series_sessions"] == 3 + 2 + 2


def test_a_date_or_timestamp_cutoff_works_like_the_string():
    frames = {"SPY": make_frame(SESSIONS)}
    a, pa = hc.trim_frames_to_history_end(frames, CUTOFF)
    b, pb = hc.trim_frames_to_history_end(frames, datetime.date(2026, 9, 21))
    pd.testing.assert_frame_equal(a["SPY"], b["SPY"])
    assert pa == pb


def test_multi_symbol_refusal_names_every_offender_in_one_error():
    frames = {
        "ZZZ": make_frame(SESSIONS[:4]),                          # ends before
        "SPY": make_frame(SESSIONS),                              # fine
        "AGG": make_frame([d for d in SESSIONS if d != CUTOFF]),  # hole on cutoff
        "IEF": make_frame(SESSIONS).iloc[0:0],                    # empty
    }
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.trim_frames_to_history_end(frames, CUTOFF)
    assert info.value.reason_code == "history_end_not_covered"
    assert info.value.symbols == ("AGG", "IEF", "ZZZ")           # sorted, complete
    for symbol in ("AGG", "IEF", "ZZZ"):
        assert symbol in str(info.value)
    assert "SPY" not in info.value.symbols


def test_multi_symbol_refusal_message_does_not_repeat_the_symbol_prefix():
    """Each per-series message already starts with its own symbol, so the
    aggregate must not prefix it a second time ('^IRX: ^IRX: no bar ...')."""
    frames = {"^IRX": make_frame([d for d in SESSIONS if d != CUTOFF]),
              "SPY": make_frame(SESSIONS)}
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.trim_frames_to_history_end(frames, CUTOFF)
    text = str(info.value)
    assert "^IRX: ^IRX:" not in text
    assert "^IRX: no bar dated exactly" in text


def test_only_empty_frames_keep_the_empty_reason_code():
    frames = {"SPY": make_frame(SESSIONS), "IEF": make_frame(SESSIONS).iloc[0:0]}
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.trim_frames_to_history_end(frames, CUTOFF)
    assert info.value.reason_code == "history_end_empty"
    assert info.value.symbols == ("IEF",)


def test_a_refusal_leaves_the_input_mapping_and_frames_untouched():
    frames = {"SPY": make_frame(SESSIONS), "IEF": make_frame(SESSIONS[:4])}
    before = {k: v.copy(deep=True) for k, v in frames.items()}
    with pytest.raises(hc.HistoryCutoffError):
        hc.trim_frames_to_history_end(frames, CUTOFF)
    assert list(frames) == ["SPY", "IEF"]
    for symbol in frames:
        pd.testing.assert_frame_equal(frames[symbol], before[symbol])


def test_no_series_at_all_is_refused_rather_than_vacuously_covered():
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.trim_frames_to_history_end({}, CUTOFF)
    assert info.value.reason_code == "history_end_not_covered"
    assert info.value.symbols == ()


def test_output_keeps_the_input_symbol_order():
    frames = {"TLT": make_frame(SESSIONS), "SPY": make_frame(SESSIONS),
              "AGG": make_frame(SESSIONS)}
    out, _ = hc.trim_frames_to_history_end(frames, CUTOFF)
    assert list(out) == ["TLT", "SPY", "AGG"]


# ------------------------------------------- bounded multi-series refusal message

BOILERPLATE = "a history end needs a real bar on that date in every series"
ENDS_BEFORE = "2026-09-15..2026-09-18"      # the history runs of SESSIONS[:4]


def _refuse(frames):
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.trim_frames_to_history_end(frames, CUTOFF)
    return info.value


def _wide(n, *, empty=False, width=3, reverse=False):
    """``n`` offending series named S000, S001, ... that all end before the
    cutoff (or, with ``empty``, hold no rows). One shared frame object is fine:
    trimming never mutates its input."""
    frame = make_frame(SESSIONS).iloc[0:0] if empty else make_frame(SESSIONS[:4])
    names = [f"S{i:0{width}d}" for i in range(n)]
    if reverse:
        names.reverse()
    return {name: frame for name in names}


def test_the_per_series_refusal_text_keeps_its_full_wording():
    """The single-series message keeps its full wording; only the aggregate is
    shortened."""
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.trim_frame_to_history_end(make_frame(SESSIONS[:4]), ts(CUTOFF), symbol="IEF")
    assert str(info.value) == (
        f"IEF: no bar dated exactly {CUTOFF} (its history runs {ENDS_BEFORE}); "
        f"{BOILERPLATE}")
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.trim_frame_to_history_end(make_frame(SESSIONS).iloc[0:0], ts(CUTOFF),
                                     symbol="SPY")
    assert str(info.value) == f"SPY: the frame has no rows to end at {CUTOFF}"


def test_a_wide_refusal_message_stays_bounded_for_500_offenders():
    exc = _refuse(_wide(500))
    assert len(str(exc)) < 2000
    assert exc.reason_code == "history_end_not_covered"
    assert str(exc).startswith(f"history end {CUTOFF} refused for 500 series")


def test_a_wide_refusal_keeps_the_symbols_tuple_complete_and_sorted():
    exc = _refuse(_wide(500, reverse=True))          # inserted in reverse order
    assert isinstance(exc.symbols, tuple) and len(exc.symbols) == 500
    assert exc.symbols == tuple(f"S{i:03d}" for i in range(500))
    assert list(exc.symbols) == sorted(exc.symbols)


def test_a_wide_refusal_lists_ten_inline_then_says_how_many_more():
    exc = _refuse(_wide(500))
    text = str(exc)
    assert "and 490 more (see symbols)" in text
    for i in range(10):                              # the first ten, sorted
        assert (f"S{i:03d}: no bar dated exactly {CUTOFF} "
                f"(history runs {ENDS_BEFORE})") in text
    for i in (10, 11, 250, 499):                     # nothing past the tenth
        assert f"S{i:03d}" not in text


def test_a_wide_refusal_states_the_shared_boilerplate_once():
    text = str(_refuse(_wide(500)))
    assert text.count(BOILERPLATE) == 1
    assert text.count("its history runs") == 0       # the aggregate omits the per-series wording


@pytest.mark.parametrize("n,more", [(10, None), (11, 1), (12, 2)])
def test_the_inline_cap_is_exactly_ten_offenders(n, more):
    text = str(_refuse(_wide(n)))
    for i in range(min(n, 10)):
        assert f"S{i:03d}: no bar dated exactly" in text
    if more is None:
        assert "more (see symbols)" not in text
    else:
        assert f"and {more} more (see symbols)" in text
        assert f"S{10:03d}" not in text


def test_a_small_refusal_still_names_each_offender_inline_with_its_own_reason():
    frames = {
        "ZZZ": make_frame(SESSIONS[:4]),                          # ends before
        "SPY": make_frame(SESSIONS),                              # fine
        "AGG": make_frame([d for d in SESSIONS if d != CUTOFF]),  # hole on cutoff
        "IEF": make_frame(SESSIONS).iloc[0:0],                    # empty
    }
    exc = _refuse(frames)
    text = str(exc)
    assert exc.symbols == ("AGG", "IEF", "ZZZ")
    assert (f"AGG: no bar dated exactly {CUTOFF} "
            "(history runs 2026-09-15..2026-09-24)") in text
    assert "IEF: the frame has no rows" in text
    assert f"ZZZ: no bar dated exactly {CUTOFF} (history runs {ENDS_BEFORE})" in text
    assert "SPY" not in text
    assert "more (see symbols)" not in text
    assert text.count(BOILERPLATE) == 1
    assert text.startswith(f"history end {CUTOFF} refused for 3 series")


def test_a_wide_refusal_of_only_empty_frames_is_bounded_and_keeps_its_code():
    exc = _refuse(_wide(500, empty=True))
    assert exc.reason_code == "history_end_empty"
    assert len(str(exc)) < 2000
    assert "S000: the frame has no rows" in str(exc)
    assert "and 490 more (see symbols)" in str(exc)
    assert len(exc.symbols) == 500


def test_a_wide_mixed_refusal_is_bounded_and_falls_back_to_not_covered():
    frames = {**_wide(250, empty=True, width=4),
              **{f"T{i:04d}": make_frame(SESSIONS[:4]) for i in range(250)}}
    exc = _refuse(frames)
    assert exc.reason_code == "history_end_not_covered"
    assert len(str(exc)) < 2000
    assert len(exc.symbols) == 500 and list(exc.symbols) == sorted(exc.symbols)


def test_a_wide_refusal_with_longer_symbols_is_still_bounded():
    frames = {f"LONGER-SYMBOL-{i:04d}": make_frame(SESSIONS[:4]) for i in range(500)}
    assert len(str(_refuse(frames))) < 2000


# ---------------------------------------------------------------- provenance

def _records():
    return {
        "SPY": {"raw_first": "2026-09-15", "raw_last": "2026-09-24",
                "kept_last": CUTOFF, "dropped_sessions": 3,
                "dropped_dates": ["2026-09-22", "2026-09-23", "2026-09-24"]},
        "IEF": {"raw_first": "2026-09-15", "raw_last": "2026-09-24",
                "kept_last": CUTOFF, "dropped_sessions": 2,
                "dropped_dates": ["2026-09-23", "2026-09-24"]},
    }


def test_provenance_shape_and_json_round_trip():
    p = hc.build_provenance(ts(CUTOFF), _records())
    assert set(p) == {"history_end", "opt_in", "decision", "by_symbol",
                      "dropped_dates", "total_dropped_series_sessions"}
    assert p["history_end"] == CUTOFF
    assert p["opt_in"] is True
    assert p["decision"] == "--history-end"
    assert list(p["by_symbol"]) == ["IEF", "SPY"]                 # sorted
    assert p["dropped_dates"] == ["2026-09-22", "2026-09-23", "2026-09-24"]
    assert p["total_dropped_series_sessions"] == 5
    assert type(p["total_dropped_series_sessions"]) is int
    assert json.loads(json.dumps(p)) == p


def test_provenance_copies_its_records():
    records = _records()
    p = hc.build_provenance(ts(CUTOFF), records)
    records["SPY"]["dropped_dates"].append("2099-01-01")
    records["SPY"]["raw_last"] = "changed"
    assert p["by_symbol"]["SPY"]["dropped_dates"] == [
        "2026-09-22", "2026-09-23", "2026-09-24"]
    assert p["by_symbol"]["SPY"]["raw_last"] == "2026-09-24"


def test_provenance_of_a_no_drop_run_has_empty_dropped_dates():
    records = {"SPY": {"raw_first": "2026-09-15", "raw_last": CUTOFF,
                       "kept_last": CUTOFF, "dropped_sessions": 0,
                       "dropped_dates": []}}
    p = hc.build_provenance(ts(CUTOFF), records)
    assert p["dropped_dates"] == []
    assert p["total_dropped_series_sessions"] == 0


def test_provenance_needs_at_least_one_series():
    with pytest.raises(ValueError):
        hc.build_provenance(ts(CUTOFF), {})


# -------------------------------------------------------------------- clause

def test_marker_constant_is_the_documented_string():
    assert hc.HISTORY_END_MARKER == "HISTORY END (opt-in"


def test_clause_states_every_required_fact():
    clause = hc.history_end_clause(hc.build_provenance(ts(CUTOFF), _records()))
    assert clause.startswith(hc.HISTORY_END_MARKER)
    assert CUTOFF in clause
    assert "--history-end" in clause
    assert "by an explicit setting" in clause
    assert "the vendor history extended past it" in clause
    assert "2026-09-22..2026-09-24" in clause                     # first..last
    assert "3 distinct dates" in clause                           # the count
    assert "IEF 2026-09-24" in clause and "SPY 2026-09-24" in clause
    assert clause.index("IEF 2026-09-24") < clause.index("SPY 2026-09-24")
    assert "Dropped sessions per series: IEF 2, SPY 3." in clause  # counts differ
    assert "The setting added, moved or repaired no bar." in clause
    assert "permanently ends at 2026-09-21" in clause
    assert "cannot be re-scored on longer history" in clause
    assert "$" not in clause
    assert "\n" not in clause


def test_clause_does_not_claim_a_holdout_exists():
    """An unscored (not_scoreable) row pins NO holdout, and this clause is
    written on that row too: it must say the scored ACCOUNT ends at the cutoff
    and that any holdout is its tail, never that a holdout exists."""
    clause = hc.history_end_clause(hc.build_provenance(ts(CUTOFF), _records()))
    assert "scored account therefore permanently ends at 2026-09-21" in clause
    assert "any ABI-1 holdout is its tail" in clause
    assert clause.count("holdout") == 1        # the conditional mention above is the only one


def test_clause_is_deterministic_and_independent_of_mapping_order():
    records = _records()
    a = hc.history_end_clause(hc.build_provenance(ts(CUTOFF), records))
    reordered = {"SPY": records["SPY"], "IEF": records["IEF"]}
    b = hc.history_end_clause(hc.build_provenance(ts(CUTOFF), reordered))
    c = hc.history_end_clause(hc.build_provenance(ts(CUTOFF), records))
    assert a == b == c


def test_clause_of_a_no_drop_run_says_so_and_still_states_permanence():
    records = {"SPY": {"raw_first": "2026-09-15", "raw_last": CUTOFF,
                       "kept_last": CUTOFF, "dropped_sessions": 0,
                       "dropped_dates": []}}
    clause = hc.history_end_clause(hc.build_provenance(ts(CUTOFF), records))
    assert clause.startswith(hc.HISTORY_END_MARKER)
    assert "none" in clause
    assert "SPY 2026-09-21" in clause
    assert "cannot be re-scored on longer history" in clause


def _many(n, raw_last="2026-09-24", prefix="SYM", width=3):
    return {f"{prefix}{i:0{width}d}": {
        "raw_first": "2010-01-04", "raw_last": raw_last, "kept_last": CUTOFF,
        "dropped_sessions": 3,
        "dropped_dates": ["2026-09-22", "2026-09-23", "2026-09-24"]}
        for i in range(n)}


def test_clause_stays_within_the_length_cap_with_many_symbols():
    for n in (1, 12, 40, 200, 1000):
        clause = hc.history_end_clause(
            hc.build_provenance(ts(CUTOFF), _many(n)))
        assert len(clause) <= 700, n
        assert clause.startswith(hc.HISTORY_END_MARKER)
        assert "cannot be re-scored on longer history" in clause, n
        assert "permanently ends at 2026-09-21" in clause, n
        assert "$" not in clause


def test_clause_names_every_symbol_in_the_grouped_form_when_it_fits():
    # 40 short tickers on one raw last date (the pooled shape): the plain
    # "SYM DATE" list is too long, the by-date grouping is not
    records = _many(40, prefix="TK", width=2)
    clause = hc.history_end_clause(hc.build_provenance(ts(CUTOFF), records))
    for symbol in records:
        assert symbol in clause
    assert "2026-09-24: TK00, TK01" in clause
    assert "series;" not in clause and "truncated" not in clause
    assert len(clause) <= 700


def test_a_big_panel_collapses_the_largest_group_but_keeps_outliers_named():
    records = _many(300)
    records["SYM007"] = dict(records["SYM007"], raw_last="2026-09-23",
                             dropped_dates=["2026-09-22", "2026-09-23"],
                             dropped_sessions=2)
    records["SYM250"] = dict(records["SYM250"], raw_last="2026-09-22",
                             dropped_dates=["2026-09-22"], dropped_sessions=1)
    clause = hc.history_end_clause(hc.build_provenance(ts(CUTOFF), records))
    assert "2026-09-24: 298 series" in clause
    assert "2026-09-23: SYM007" in clause
    assert "2026-09-22: SYM250" in clause
    assert "SYM100" not in clause                 # the modal group is a count
    assert "truncated" not in clause
    assert len(clause) <= 700


def test_clause_truncates_visibly_rather_than_silently_when_it_must():
    # every series ends on its own date, so no grouping can shorten the list
    dates = [f"{d:%Y-%m-%d}" for d in pd.bdate_range("2025-01-01", periods=300)]
    records = {}
    for i, last in enumerate(dates):
        records[f"SYM{i:03d}"] = {
            "raw_first": "2010-01-04", "raw_last": last, "kept_last": CUTOFF,
            "dropped_sessions": 1, "dropped_dates": ["2026-09-22"]}
    clause = hc.history_end_clause(hc.build_provenance(ts(CUTOFF), records))
    assert "(truncated)" in clause
    assert len(clause) <= 700
    assert clause.startswith(hc.HISTORY_END_MARKER)
    assert "cannot be re-scored on longer history" in clause


def test_clause_distinguishes_series_with_different_last_bars():
    # eight series: the clause also carries a per-series dropped-sessions
    # sentence (this record set has differing counts), which shares the
    # 700-character cap, so twelve series would not fit the plain "SYM DATE"
    # form of the raw-last-bar listing
    records = _many(8)
    records["SYM007"] = dict(records["SYM007"], raw_last="2026-09-23",
                             dropped_dates=["2026-09-22", "2026-09-23"],
                             dropped_sessions=2)
    clause = hc.history_end_clause(hc.build_provenance(ts(CUTOFF), records))
    assert "SYM007 2026-09-23" in clause
    assert "SYM005 2026-09-24" in clause and "SYM006 2026-09-24" in clause
    assert len(clause) <= 700


# ------------------------------------- clause: wording and per-series drop counts

HEAD_EXTENDED = ("HISTORY END (opt-in setting, --history-end): the scored "
                 "history was ended at 2026-09-21 by an explicit setting; the "
                 "vendor history extended past it.")
HEAD_ENDED_THERE = ("HISTORY END (opt-in setting, --history-end): the scored "
                    "history was ended at 2026-09-21 by an explicit setting; the "
                    "vendor history ended there.")


def _rec(k, *, first="2010-01-04"):
    """A record for a series that lost ``k`` sessions after the cutoff: its
    dropped dates are the ``k`` business days after 2026-09-21, so a bigger k
    also means a later raw last bar."""
    dates = [f"{d:%Y-%m-%d}" for d in pd.bdate_range("2026-09-22", periods=k)]
    return {"raw_first": first, "raw_last": dates[-1] if dates else CUTOFF,
            "kept_last": CUTOFF, "dropped_sessions": k, "dropped_dates": dates}


def _panel(counts):
    """Series SYM000.. with the given per-series drop counts."""
    return {f"SYM{i:03d}": _rec(k) for i, k in enumerate(counts)}


def _clause(records):
    return hc.history_end_clause(hc.build_provenance(ts(CUTOFF), records))


def test_clause_head_says_the_vendor_history_extended_past_the_cutoff():
    clause = _clause(_records())
    assert clause.startswith(HEAD_EXTENDED)
    assert clause.startswith(hc.HISTORY_END_MARKER)


def test_clause_head_says_the_vendor_history_ended_there_when_nothing_dropped():
    clause = _clause({"SPY": _rec(0)})
    assert clause.startswith(HEAD_ENDED_THERE)
    assert "extended past it" not in clause


def test_clause_states_the_reason_is_an_explicit_setting():
    for records in (_records(), {"SPY": _rec(0)}, _panel([1, 2, 3])):
        clause = _clause(records)
        assert "by an explicit setting" in clause
        assert "The setting added, moved or repaired no bar." in clause


def test_clause_states_the_setting_added_moved_or_repaired_no_bar():
    for records in (_records(), {"SPY": _rec(0)}, _many(300)):
        clause = _clause(records)
        assert "The setting added, moved or repaired no bar." in clause
        assert "permanently ends at 2026-09-21" in clause
        assert "any ABI-1 holdout is its tail" in clause
        assert "cannot be re-scored on longer history" in clause


def test_clause_lists_drop_counts_per_series_only_when_they_differ():
    same = _clause(_panel([3, 3, 3, 3]))
    assert "Dropped sessions per series" not in same
    differ = _clause(_records())                       # SPY 3, IEF 2
    assert "Dropped sessions per series: IEF 2, SPY 3." in differ
    # order: the dropped-dates statement, then the per-series counts, then the
    # raw last bars, then the fixed tail
    order = [differ.index(part) for part in (
        "Later dates dropped:", "Dropped sessions per series:", "Raw last bars:",
        "The setting added, moved or repaired no bar.")]
    assert order == sorted(order)


def test_a_single_series_never_gets_a_drop_count_sentence():
    assert "Dropped sessions per series" not in _clause({"SPY": _rec(3)})


def test_a_series_that_lost_nothing_makes_the_counts_differ():
    clause = _clause({"SPY": _rec(3), "IEF": _rec(0)})
    assert "Dropped sessions per series: IEF 0, SPY 3." in clause


def test_a_big_panel_collapses_the_largest_count_group_but_keeps_outliers_named():
    counts = [3] * 300
    counts[7], counts[250] = 2, 1
    clause = _clause(_panel(counts))
    assert "Dropped sessions per series: " in clause
    assert "3: 298 series" in clause
    assert "2: SYM007" in clause
    assert "1: SYM250" in clause
    assert "SYM100 3" not in clause
    assert len(clause) <= hc.CLAUSE_MAX_CHARS


def test_drop_counts_group_by_count_in_numeric_order_when_the_plain_form_is_long():
    # 20 short-named series over three counts: the plain "SYM n" list is too
    # long for the drop-count share of the cap, the grouped form is not, and the
    # groups run 9, 10, 100 numerically (as text they would run 10, 100, 9)
    records = {f"A{i:02d}": _rec([9, 10, 100][i % 3]) for i in range(20)}
    clause = _clause(records)
    part = clause.split("Dropped sessions per series: ")[1].split(" Raw last bars: ")[0]
    assert part == ("9: A00, A03, A06, A09, A12, A15, A18; "
                    "10: A01, A04, A07, A10, A13, A16, A19; "
                    "100: A02, A05, A08, A11, A14, A17.")
    assert len(clause) <= hc.CLAUSE_MAX_CHARS


@pytest.mark.parametrize("n", [2, 3, 12, 40, 99, 100, 101, 300, 1000])
@pytest.mark.parametrize("shape", ["distinct", "few_counts", "one_outlier"])
def test_clause_stays_within_the_cap_for_any_number_of_series(n, shape):
    if shape == "distinct":                            # worst case: every count
        counts = [i + 1 for i in range(n)]             # and every last bar differs
    elif shape == "few_counts":
        counts = [(i % 4) + 1 for i in range(n)]
    else:
        counts = [3] * n
        counts[n // 2] = 5
    clause = _clause(_panel(counts))
    assert len(clause) <= hc.CLAUSE_MAX_CHARS, (n, shape, len(clause))
    assert clause.startswith(HEAD_EXTENDED)
    assert "The setting added, moved or repaired no bar." in clause
    assert "permanently ends at 2026-09-21" in clause
    assert "cannot be re-scored on longer history" in clause
    assert "\n" not in clause and "$" not in clause


def test_the_hundred_series_worst_case_is_truncated_visibly_not_silently():
    clause = _clause(_panel([i + 1 for i in range(100)]))
    assert len(clause) <= hc.CLAUSE_MAX_CHARS
    assert "(truncated)" in clause
    assert "Dropped sessions per series: " in clause and "Raw last bars: " in clause


def test_clause_with_drop_counts_is_deterministic_and_order_independent():
    counts = [(i * 7) % 11 + 1 for i in range(100)]
    records = _panel(counts)
    forward = _clause(records)
    backward = _clause(dict(reversed(list(records.items()))))
    again = _clause(records)
    assert forward == backward == again


def test_clause_drop_and_raw_last_listings_share_the_cap_without_starving_either():
    counts = [1] * 50 + [2] * 50
    clause = _clause(_panel(counts))
    drop_part = clause.split("Dropped sessions per series: ")[1].split(" Raw last bars: ")[0]
    raw_part = clause.split("Raw last bars: ")[1].split(" The setting added")[0]
    assert "SYM000" in drop_part or "1:" in drop_part
    assert "SYM000" in raw_part or "2026-09-22" in raw_part
    assert len(clause) <= hc.CLAUSE_MAX_CHARS


def test_provenance_input_is_not_mutated_by_the_clause():
    p = hc.build_provenance(ts(CUTOFF), _records())
    before = json.dumps(p, sort_keys=True)
    hc.history_end_clause(p)
    assert json.dumps(p, sort_keys=True) == before


def test_module_docs_do_not_claim_the_holdout_end_is_always_pinned():
    for text in (hc.__doc__, hc.history_end_clause.__doc__):
        assert "pins none" in " ".join(text.split())   # a not-scoreable row pins no holdout
    assert "first_test" in hc.__doc__          # states the conditional pin
    assert "The pin is not unconditional" in " ".join(hc.__doc__.split())


# ------------------------------------------------------- real-run guard

def _prov(counts, cutoff=CUTOFF):
    return hc.build_provenance(ts(cutoff), _panel(counts))


def test_guard_constants_and_reason_codes():
    assert hc.MAX_UNACCEPTED_DROP_SESSIONS == 10
    assert hc.REASON_AFTER_TESTED_DATE == "history_end_after_tested_date"
    assert hc.REASON_DROP_TOO_LARGE == "history_end_drop_too_large"


def test_guard_passes_a_normal_run_and_returns_none():
    assert hc.check_real_run_guard(_prov([3, 2, 0]), tested_date="2026-09-25") is None


def test_guard_allows_a_cutoff_equal_to_the_tested_date():
    assert hc.check_real_run_guard(_prov([0]), tested_date=CUTOFF) is None


def test_guard_refuses_a_cutoff_later_than_the_tested_date():
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.check_real_run_guard(_prov([0]), tested_date="2026-09-18")
    assert info.value.reason_code == "history_end_after_tested_date"
    assert "2026-09-21" in str(info.value) and "2026-09-18" in str(info.value)
    assert isinstance(info.value, ValueError)


@pytest.mark.parametrize("bad", [
    "", " ", "2026-9-25", "2026/09/25", "2026-09-25T00:00:00", "2026-02-30",
    "today", None, int("2026" + "0925"), ["2026-09-25"],
], ids=["empty", "space", "unpadded", "slashes", "with_time", "impossible_day", "word", "none",
        "compact_integer", "list"])
def test_guard_refuses_an_unparseable_tested_date_as_invalid(bad):
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.check_real_run_guard(_prov([0]), tested_date=bad)
    assert info.value.reason_code == "history_end_invalid"
    assert "tested_date" in str(info.value)


def test_guard_accepts_a_date_object_for_tested_date():
    assert hc.check_real_run_guard(
        _prov([0]), tested_date=datetime.date(2026, 9, 25)) is None


def test_guard_default_allows_exactly_ten_dropped_sessions_and_refuses_eleven():
    assert hc.check_real_run_guard(_prov([10, 1]), tested_date="2026-09-25") is None
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.check_real_run_guard(_prov([11, 1]), tested_date="2026-09-25")
    assert info.value.reason_code == "history_end_drop_too_large"


def test_guard_drop_refusal_names_the_worst_series_and_the_way_out():
    prov = hc.build_provenance(ts(CUTOFF), {
        "AAA": _rec(12), "SPY": _rec(40), "ZZZ": _rec(40), "IEF": _rec(2)})
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.check_real_run_guard(prov, tested_date="2026-09-25")
    text = str(info.value)
    assert info.value.reason_code == "history_end_drop_too_large"
    assert "SPY" in text and "40" in text          # worst; ties -> first by symbol
    assert "pass --max-history-drop N to accept a larger drop" in text
    assert info.value.symbols == ("AAA", "SPY", "ZZZ")      # every offender, sorted


def test_guard_max_drop_is_the_callers_explicit_acceptance():
    prov = _prov([40, 2])
    assert hc.check_real_run_guard(prov, tested_date="2026-09-25", max_drop=40) is None
    with pytest.raises(hc.HistoryCutoffError):
        hc.check_real_run_guard(prov, tested_date="2026-09-25", max_drop=39)
    assert hc.check_real_run_guard(
        _prov([0, 0]), tested_date="2026-09-25", max_drop=0) is None
    with pytest.raises(hc.HistoryCutoffError):
        hc.check_real_run_guard(_prov([1]), tested_date="2026-09-25", max_drop=0)


@pytest.mark.parametrize("bad", [-1, 1.5, "10", None, True, False, [10]],
                         ids=lambda v: repr(v))
def test_guard_refuses_a_bad_max_drop_as_invalid(bad):
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.check_real_run_guard(_prov([1]), tested_date="2026-09-25", max_drop=bad)
    assert info.value.reason_code == "history_end_invalid"
    assert "max_drop" in str(info.value)


def test_guard_checks_the_tested_date_before_the_drop_size():
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.check_real_run_guard(_prov([50]), tested_date="2026-09-18")
    assert info.value.reason_code == "history_end_after_tested_date"


def test_guard_with_no_provenance_is_off_and_looks_at_nothing():
    # default (flag unset): provenance is None and the guard must not even
    # validate the other arguments
    assert hc.check_real_run_guard(None, tested_date="not a date", max_drop=-5) is None


def test_guard_does_not_mutate_the_provenance():
    prov = _prov([3, 2])
    before = json.dumps(prov, sort_keys=True)
    hc.check_real_run_guard(prov, tested_date="2026-09-25")
    assert json.dumps(prov, sort_keys=True) == before


# --------------------------------------- door validator (runs before any load)

BAD_TESTED_DATES = [
    "", " ", "2026-9-25", "2026/09/25", "2026-09-25T00:00:00", "2026-02-30",
    "today", None, int("2026" + "0925"), ["2026-09-25"],
    datetime.datetime(2026, 9, 25, 13, 30), 20.5,
]
BAD_TESTED_DATE_IDS = ["empty", "space", "unpadded", "slashes", "with_time", "impossible_day", "word", "none",
                       "compact_integer", "list", "datetime", "float"]
BAD_MAX_DROPS = [-1, 1.5, "10", None, True, False, [10]]


def test_validate_guard_arguments_accepts_good_arguments_and_returns_none():
    assert hc.validate_guard_arguments(tested_date="2026-09-25", max_drop=10) is None
    assert hc.validate_guard_arguments(
        tested_date=datetime.date(2026, 9, 25), max_drop=0) is None
    assert hc.validate_guard_arguments(
        tested_date=hc.parse_history_end(CUTOFF), max_drop=250) is None


def test_validate_guard_arguments_is_keyword_only_and_needs_both():
    with pytest.raises(TypeError):
        hc.validate_guard_arguments("2026-09-25", 10)
    with pytest.raises(TypeError):
        hc.validate_guard_arguments(tested_date="2026-09-25")
    with pytest.raises(TypeError):
        hc.validate_guard_arguments(max_drop=10)


@pytest.mark.parametrize("bad", BAD_TESTED_DATES, ids=BAD_TESTED_DATE_IDS)
def test_validate_guard_arguments_refuses_every_bad_tested_date(bad):
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.validate_guard_arguments(tested_date=bad, max_drop=10)
    assert info.value.reason_code == "history_end_invalid"
    assert "tested_date" in str(info.value)
    assert isinstance(info.value, ValueError)


@pytest.mark.parametrize("bad", BAD_MAX_DROPS, ids=lambda v: repr(v))
def test_validate_guard_arguments_refuses_every_bad_max_drop(bad):
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.validate_guard_arguments(tested_date="2026-09-25", max_drop=bad)
    assert info.value.reason_code == "history_end_invalid"
    assert "max_drop" in str(info.value)


@pytest.mark.parametrize("bad", BAD_TESTED_DATES, ids=BAD_TESTED_DATE_IDS)
def test_the_guard_and_the_door_validator_say_the_same_thing_for_a_bad_tested_date(bad):
    with pytest.raises(hc.HistoryCutoffError) as door:
        hc.validate_guard_arguments(tested_date=bad, max_drop=10)
    with pytest.raises(hc.HistoryCutoffError) as guard:
        hc.check_real_run_guard(_prov([0]), tested_date=bad)
    assert guard.value.reason_code == door.value.reason_code == "history_end_invalid"
    assert str(guard.value) == str(door.value)


@pytest.mark.parametrize("bad", BAD_MAX_DROPS, ids=lambda v: repr(v))
def test_the_guard_and_the_door_validator_say_the_same_thing_for_a_bad_max_drop(bad):
    with pytest.raises(hc.HistoryCutoffError) as door:
        hc.validate_guard_arguments(tested_date="2026-09-25", max_drop=bad)
    with pytest.raises(hc.HistoryCutoffError) as guard:
        hc.check_real_run_guard(_prov([1]), tested_date="2026-09-25", max_drop=bad)
    assert guard.value.reason_code == door.value.reason_code == "history_end_invalid"
    assert str(guard.value) == str(door.value)


def test_validate_guard_arguments_checks_max_drop_before_tested_date_like_the_guard():
    with pytest.raises(hc.HistoryCutoffError) as door:
        hc.validate_guard_arguments(tested_date="nope", max_drop=-1)
    with pytest.raises(hc.HistoryCutoffError) as guard:
        hc.check_real_run_guard(_prov([1]), tested_date="nope", max_drop=-1)
    assert "max_drop" in str(door.value) and "tested_date" not in str(door.value)
    assert str(door.value) == str(guard.value)


def test_validate_guard_arguments_is_not_switched_off_by_a_missing_provenance():
    # the guard is a no-op with provenance None; the door validator has no
    # provenance and always validates (drivers call it whenever a history end is set)
    assert hc.check_real_run_guard(None, tested_date="not a date", max_drop=-5) is None
    with pytest.raises(hc.HistoryCutoffError):
        hc.validate_guard_arguments(tested_date="not a date", max_drop=-5)


def test_the_guard_calls_the_door_validator_and_only_when_a_history_end_is_set(
        monkeypatch):
    seen = []

    def spy(*, tested_date, max_drop):
        seen.append((tested_date, max_drop))
        raise hc.HistoryCutoffError(hc.REASON_INVALID, "from the door validator")

    monkeypatch.setattr(hc, "validate_guard_arguments", spy)
    assert hc.check_real_run_guard(None, tested_date="2026-09-25", max_drop=4) is None
    assert seen == []                                     # off: never consulted
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.check_real_run_guard(_prov([0]), tested_date="2026-09-25", max_drop=4)
    assert str(info.value) == "from the door validator"
    assert seen == [("2026-09-25", 4)]


def test_a_valid_argument_pair_leaves_the_guard_behaviour_unchanged():
    assert hc.check_real_run_guard(_prov([3, 2, 0]), tested_date="2026-09-25",
                                   max_drop=3) is None
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.check_real_run_guard(_prov([11]), tested_date="2026-09-25")
    assert info.value.reason_code == "history_end_drop_too_large"


# ------------------------------------------------ reason codes + refusal payload

ALL_SIX = ("history_end_invalid", "history_end_not_a_session",
           "history_end_not_covered", "history_end_empty",
           "history_end_after_tested_date", "history_end_drop_too_large")
CACHE_SENTENCE = ("The per-run OHLC cache written before this check is kept (no "
                  "ledger row, result file or report is written); retry with a "
                  "fresh --run-id: a reused run-id reads that cached history "
                  "instead of fetching again (real runs of the bespoke overlay "
                  "drivers refuse it as stale_ohlc_cache).")


def test_reason_code_tuple_holds_all_six_codes():
    assert isinstance(hc.HISTORY_END_REASON_CODES, tuple)
    assert set(hc.HISTORY_END_REASON_CODES) == set(ALL_SIX)
    assert len(hc.HISTORY_END_REASON_CODES) == 6
    assert set(hc.HISTORY_END_REASON_CODES) == {
        hc.REASON_INVALID, hc.REASON_NOT_A_SESSION, hc.REASON_NOT_COVERED,
        hc.REASON_EMPTY, hc.REASON_AFTER_TESTED_DATE, hc.REASON_DROP_TOO_LARGE}


def test_every_code_the_module_can_raise_is_in_the_tuple():
    raised = set()
    frames = {"SPY": make_frame(SESSIONS), "IEF": make_frame(SESSIONS[:4]),
              "EMP": make_frame(SESSIONS).iloc[0:0]}
    attempts = [
        lambda: hc.parse_history_end("2026-9-21"),
        lambda: hc.parse_history_end("2026-09-19"),
        lambda: hc.trim_frames_to_history_end(frames, CUTOFF),
        lambda: hc.trim_frames_to_history_end({"EMP": frames["EMP"]}, CUTOFF),
        lambda: hc.check_real_run_guard(_prov([0]), tested_date="2026-09-18"),
        lambda: hc.check_real_run_guard(_prov([99]), tested_date="2026-09-25"),
    ]
    for attempt in attempts:
        with pytest.raises(hc.HistoryCutoffError) as info:
            attempt()
        raised.add(info.value.reason_code)
    assert raised == set(ALL_SIX)
    assert raised <= set(hc.HISTORY_END_REASON_CODES)


def _refused(code, message="the message", symbols=()):
    return hc.HistoryCutoffError(code, message, symbols=symbols)


def test_refusal_payload_has_the_standard_shape_and_key_order():
    exc = _refused("history_end_invalid", "history end must be a date", symbols=())
    payload = hc.refusal_payload(exc, history_end_requested="2026-9-21")
    assert list(payload) == ["reason_code", "reason", "history_end_requested",
                             "symbols"]
    assert payload == {"reason_code": "history_end_invalid",
                       "reason": "history end must be a date",
                       "history_end_requested": "2026-9-21", "symbols": []}
    assert type(payload["symbols"]) is list
    json.dumps(payload)


def test_refusal_payload_merges_extra_fields_after_the_standard_ones():
    exc = _refused("history_end_invalid", "m")
    payload = hc.refusal_payload(exc, history_end_requested=None,
                                 run_id="r1", dry_run=True)
    assert list(payload) == ["reason_code", "reason", "history_end_requested",
                             "symbols", "run_id", "dry_run"]
    assert payload["history_end_requested"] is None
    assert payload["run_id"] == "r1" and payload["dry_run"] is True


@pytest.mark.parametrize("code", [
    "history_end_not_covered", "history_end_empty",
    "history_end_after_tested_date", "history_end_drop_too_large"])
def test_refusal_payload_adds_the_cache_kept_sentence_for_every_post_load_refusal(code):
    """All four are raised AFTER the load step wrote the per-run OHLC cache
    (the two guard refusals need the provenance the trim produces)."""
    exc = _refused(code, "IEF: no bar dated exactly 2026-09-21", symbols=("IEF",))
    payload = hc.refusal_payload(exc, history_end_requested=CUTOFF)
    assert payload["reason"].endswith(CACHE_SENTENCE)
    assert payload["reason"].startswith("IEF: no bar dated exactly 2026-09-21")
    assert payload["symbols"] == ["IEF"]
    assert payload["reason_code"] == code


def test_refusal_payload_does_not_double_a_terminal_period():
    exc = _refused("history_end_empty", "SPY: the frame has no rows.")
    payload = hc.refusal_payload(exc, history_end_requested=CUTOFF)
    assert payload["reason"] == f"SPY: the frame has no rows. {CACHE_SENTENCE}"
    assert ".." not in payload["reason"]


@pytest.mark.parametrize("code", ["history_end_invalid", "history_end_not_a_session"])
def test_refusal_payload_reason_is_just_the_message_for_door_refusals(code):
    exc = _refused(code, "a plain message with no retry advice")
    payload = hc.refusal_payload(exc, history_end_requested=CUTOFF)
    assert payload["reason"] == "a plain message with no retry advice"
    assert "cache" not in payload["reason"] and "--run-id" not in payload["reason"]


def test_refusal_payload_history_end_requested_is_the_string_or_none():
    exc = _refused("history_end_invalid", "m")
    assert hc.refusal_payload(exc, history_end_requested=None)[
        "history_end_requested"] is None
    assert hc.refusal_payload(exc, history_end_requested="2026-09-21")[
        "history_end_requested"] == "2026-09-21"
    assert hc.refusal_payload(exc, history_end_requested=datetime.date(2026, 9, 21))[
        "history_end_requested"] == "2026-09-21"


def test_refusal_payload_from_a_real_trim_refusal_carries_the_symbols():
    frames = {"SPY": make_frame(SESSIONS), "IEF": make_frame(SESSIONS[:4])}
    with pytest.raises(hc.HistoryCutoffError) as info:
        hc.trim_frames_to_history_end(frames, CUTOFF)
    payload = hc.refusal_payload(info.value, history_end_requested=CUTOFF)
    assert payload["reason_code"] == "history_end_not_covered"
    assert payload["symbols"] == ["IEF"]
    assert payload["reason"].endswith(CACHE_SENTENCE)


def test_refusal_payload_needs_a_history_cutoff_error():
    with pytest.raises(TypeError):
        hc.refusal_payload(ValueError("plain"), history_end_requested=None)
    with pytest.raises(TypeError):
        hc.refusal_payload(None, history_end_requested=None)


def test_refusal_payload_history_end_requested_is_keyword_only_and_required():
    exc = _refused("history_end_invalid", "m")
    with pytest.raises(TypeError):
        hc.refusal_payload(exc, CUTOFF)
    with pytest.raises(TypeError):
        hc.refusal_payload(exc)
