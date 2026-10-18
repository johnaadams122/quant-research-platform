import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import datetime as dt
import pytest
import trading_calendar as pc

ARTIFACT = os.path.join(os.path.dirname(__file__), "..", "..",
                        "data", "live", "nyse_calendar_2026_2028.json")
SHA = "3127f0fd32be80fbc6c2919c85ce0c3d718ff7e9bbe9b0dd41a495062a97007e"

def test_load_verifies_sha_and_parses_dates():
    s = pc.load_sessions(ARTIFACT, SHA)
    assert dt.date(2026, 1, 2) in s
    assert dt.date(2026, 7, 2) in s            # July 2 IS a session
    assert dt.date(2026, 7, 3) not in s        # observed Independence Day CLOSED
    assert dt.date(2026, 12, 25) not in s      # Christmas
    assert all(isinstance(x, dt.date) for x in s)

def test_representative_holidays_absent_and_endpoints_sorted_unique():
    s = pc.load_sessions(ARTIFACT, SHA)
    assert dt.date(2026, 4, 3) not in s        # Good Friday 2026
    assert dt.date(2026, 1, 19) not in s       # MLK Day 2026
    assert dt.date(2026, 6, 19) not in s       # Juneteenth 2026
    assert s == sorted(s) and len(s) == len(set(s))
    assert s[0] == dt.date(2026, 1, 2)

def test_sha_mismatch_raises():
    with pytest.raises(ValueError):
        pc.load_sessions(ARTIFACT, "deadbeef")

def test_assert_covered_fails_closed_past_end():
    s = pc.load_sessions(ARTIFACT, SHA)
    pc.assert_covered(dt.date(2027, 1, 4), s)
    with pytest.raises(ValueError):
        pc.assert_covered(dt.date(2029, 1, 2), s)

# --- authoritative constants + default accessor ---

def test_constants_live_in_trading_calendar_with_pinned_values():
    # trading_calendar is the single authoritative home for the NYSE pin;
    # other modules import it from here rather than keeping their own copy.
    assert pc.NYSE_CAL_REL == "data/live/nyse_calendar_2026_2028.json"
    assert pc.NYSE_CAL_SHA256 == SHA

def test_default_calendar_path_derives_from_module_file_never_cwd(tmp_path, monkeypatch):
    expected = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(pc.__file__)), "..",
        *"data/live/nyse_calendar_2026_2028.json".split("/")))
    monkeypatch.chdir(tmp_path)  # a cwd-derived path would break here
    assert os.path.abspath(pc.default_calendar_path()) == expected

def test_load_default_sessions_roundtrips_real_artifact_and_pin():
    s = pc.load_default_sessions()
    assert s[0] == dt.date(2026, 1, 2)
    assert s[-1] == dt.date(2028, 12, 29)
    assert len(s) == 753
    assert dt.date(2026, 9, 7) not in s        # Labor Day 2026

# --- Saturday New Year's Day must NOT close the preceding Friday ---

def test_saturday_new_years_day_does_not_close_the_preceding_friday():
    # NYSE footnote: "Because the holiday falls on Saturday, January 1, 2028, no
    # New Year's Day holiday is observed." The exchange was OPEN Fri 2004-12-31,
    # 2010-12-31 and 2021-12-31 (same calendar shape). pandas'
    # USFederalHolidayCalendar is not a safe generator here: its
    # nearest_workday observance moves a Saturday Jan 1 back to Friday Dec 31
    # and would wrongly close 2027-12-31 -- while Independence Day and
    # Christmas on a Saturday DO close the preceding Friday on the NYSE (below).
    s = pc.load_default_sessions()
    assert dt.date(2027, 12, 31) in s          # Friday, open
    assert dt.date(2028, 1, 1) not in s        # Saturday
    assert dt.date(2028, 1, 3) in s            # first session of 2028
    assert dt.date(2027, 12, 24) not in s      # Christmas (Sat 12-25) observed Friday
    assert dt.date(2026, 7, 3) not in s        # Independence Day (Sat 07-04) observed Friday

def test_vendored_calendar_matches_exchange_calendars_xnys_over_its_whole_span():
    # Whole-set cross-check (not just spot dates) against an independent
    # implementation. exchange_calendars is a dev-env convenience, not a
    # runtime dependency of the stdlib-only launcher probe, so skip if absent.
    xc = pytest.importorskip("exchange_calendars")
    pd = pytest.importorskip("pandas")
    s = pc.load_default_sessions()
    cal = xc.get_calendar("XNYS", start="2025-01-01", end="2029-12-31")
    expected = [d.date() for d in cal.sessions_in_range(
        pd.Timestamp(s[0]), pd.Timestamp(s[-1]))]
    assert s == expected

def test_pin_holds_for_lf_and_crlf_copies_and_rejects_edits(tmp_path):
    import trading_calendar as tc
    original = open(tc.default_calendar_path(), "rb").read().replace(b"\r\n", b"\n")
    lf = tmp_path / "lf.json"
    crlf = tmp_path / "crlf.json"
    edited = tmp_path / "edited.json"
    lf.write_bytes(original)
    crlf.write_bytes(original.replace(b"\n", b"\r\n"))
    edited.write_bytes(original.replace(b"2026-", b"2025-", 1))
    assert tc.load_sessions(str(lf), tc.NYSE_CAL_SHA256) == tc.load_sessions(str(crlf), tc.NYSE_CAL_SHA256)
    with pytest.raises(ValueError, match="sha mismatch"):
        tc.load_sessions(str(edited), tc.NYSE_CAL_SHA256)
