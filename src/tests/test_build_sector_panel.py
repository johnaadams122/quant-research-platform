"""Guards for scripts/build_sector_panel.py, the disk-facing shell around
src/mining/panel.py.

No network anywhere: every fetch below is a synthetic fetcher that returns a
frame shaped like data_pipeline.fetch_ohlc's output (a ``Date`` column of
``datetime.date`` objects plus Open/High/Low/Close), and every expectation is
hand-countable via the same monotone-ramp trick test_mining_panel.py uses.
The one on-disk artifact read here is the committed membership file, which
is a static hand-dated JSON document rather than fetched data.
"""
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ROOT = os.path.abspath(os.path.join(_SRC, ".."))
_SCRIPTS = os.path.join(_ROOT, "scripts")
for _path in (_SRC, _SCRIPTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mining import constants  # noqa: E402
from mining import panel  # noqa: E402
import build_sector_panel as bsp  # noqa: E402


BUILT_AT = datetime(2026, 9, 4, 21, 0, 0, tzinfo=timezone.utc)

EXPECTED_MEMBERS = {
    "XLB": "1998-12-22", "XLE": "1998-12-22", "XLF": "1998-12-22",
    "XLI": "1998-12-22", "XLK": "1998-12-22", "XLP": "1998-12-22",
    "XLU": "1998-12-22", "XLV": "1998-12-22", "XLY": "1998-12-22",
    "XLRE": "2015-10-08", "XLC": "2018-06-19",
}


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

def _sessions(n, start="2020-01-01"):
    return pd.DatetimeIndex(pd.bdate_range(start=start, periods=n))


def _ramp(sessions, first, step, *, drop=()):
    """Strictly monotone closes: above every trailing own-mean if step > 0,
    below it if step < 0, so every fraction below is an exact ratio."""
    values = [float(first) + step * i for i in range(len(sessions))]
    series = pd.Series(values, index=pd.DatetimeIndex(sessions), dtype=float)
    if drop:
        series = series.drop(index=[sessions[i] for i in drop])
    return series


def _ohlc_frame(closes: pd.Series) -> pd.DataFrame:
    """The shape data_pipeline.fetch_ohlc returns: a Date column of dates."""
    return pd.DataFrame({
        "Date": [stamp.date() for stamp in closes.index],
        "Open": closes.to_numpy() - 0.5,
        "High": closes.to_numpy() + 1.0,
        "Low": closes.to_numpy() - 1.0,
        "Close": closes.to_numpy(),
    })


def _membership_abc(start="2020-01-01"):
    return bsp.build_membership_dict(
        [("A", start, None), ("B", start, None), ("C", start, None)],
        basis_id="test_basis", notes="synthetic")


def _fetcher_from(frames: dict):
    """A fetcher returning fetch_ohlc-shaped frames; None for unknown names."""
    def fetch(ticker):
        closes = frames.get(ticker)
        return None if closes is None else _ohlc_frame(closes)
    return fetch


def _abc_frames(n=12):
    sessions = _sessions(n)
    return {
        "A": _ramp(sessions, 100.0, 1.0),
        "B": _ramp(sessions, 200.0, 1.0),
        "C": _ramp(sessions, 500.0, -0.05),
    }


def _build_abc_panel(tmp_path, n=12):
    frames = _abc_frames(n)
    membership = _membership_abc()
    membership_path = tmp_path / "membership.json"
    bsp.write_membership_file(str(membership_path), membership)
    cache_dir = tmp_path / "cache"
    bsp.write_cache(str(cache_dir), list(frames), fetcher=_fetcher_from(frames),
                    basis_id="test_basis", membership=membership,
                    membership_path=str(membership_path), fetch_date="2026-09-04")
    return bsp.build_sector_panel(
        cache_dir=str(cache_dir), membership_path=str(membership_path))


# --------------------------------------------------------------------------
# (a) the committed membership file
# --------------------------------------------------------------------------

def test_committed_membership_file_loads_with_exactly_the_eleven_dated_members():
    path = bsp.DEFAULT_MEMBERSHIP_PATH
    assert os.path.exists(path), path
    membership = panel.load_membership(path)
    assert membership["basis_id"] == "sector_spdr_11"
    assert membership["schema_version"] == 1
    assert membership["point_in_time"] is True
    assert membership["ledgerable"] is True
    assert membership["ledger_block_reason"] is None
    assert isinstance(membership.get("notes"), str) and membership["notes"]
    members = {entry["ticker"]: entry for entry in membership["members"]}
    assert set(members) == set(EXPECTED_MEMBERS)
    assert len(membership["members"]) == 11
    for ticker, start in EXPECTED_MEMBERS.items():
        assert members[ticker]["start"] == start, ticker
        assert members[ticker]["end"] is None, ticker


def test_build_membership_dict_matches_the_committed_file_member_for_member():
    built = bsp.build_membership_dict(notes="x")
    assert [(m["ticker"], m["start"], m["end"]) for m in built["members"]] == [
        (t, s, e) for t, s, e in bsp.SECTOR_SPDR_11_MEMBERS]
    assert built["members"][0]["ticker"] == "XLB"


# --------------------------------------------------------------------------
# (b) cache + manifest round-trip, tamper rejection
# --------------------------------------------------------------------------

def test_write_cache_round_trips_through_load_cached_closes(tmp_path):
    frames = _abc_frames(8)
    cache_dir = tmp_path / "cache"
    manifest = bsp.write_cache(
        str(cache_dir), ["C", "A", "B"], fetcher=_fetcher_from(frames),
        basis_id="test_basis", fetch_date="2026-09-04")

    manifest_path = bsp.manifest_path_for(str(cache_dir))
    assert os.path.basename(manifest_path) == panel.CACHE_SEAL_FILENAME
    assert os.path.exists(manifest_path)
    on_disk = panel.load_panel_manifest(manifest_path)
    assert on_disk == manifest
    assert manifest["adjustment"] == constants.PANEL_PRICE_ADJUSTMENT
    assert manifest["failed"] == {}
    assert manifest["tickers_cached"] == ["A", "B", "C"]
    for ticker in "ABC":
        entry = manifest["per_ticker"][ticker]
        assert set(entry) == {"path", "sha256", "rows", "first_date",
                              "last_date"}
        assert entry["rows"] == 8
        assert entry["first_date"] == "2020-01-01"
        raw = (cache_dir / entry["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    assert manifest["panel_fingerprint"] == bsp.panel_fingerprint(manifest)
    assert len(manifest["panel_fingerprint"]) == 64

    # The CSV keeps Open/High/Low too, with Date as the first column.
    header = (cache_dir / "A.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "Date,Open,High,Low,Close"

    loaded = panel.load_cached_closes(
        str(cache_dir), ["A", "B", "C"], manifest_path=manifest_path)
    assert set(loaded) == {"A", "B", "C"}
    expected = frames["C"].rename("C").rename_axis("Date")
    pd.testing.assert_series_equal(
        loaded["C"], expected, check_names=True, check_index_type=False,
        check_freq=False)


def test_a_tampered_csv_is_rejected_by_the_loader(tmp_path):
    frames = _abc_frames(8)
    cache_dir = tmp_path / "cache"
    bsp.write_cache(str(cache_dir), list(frames), fetcher=_fetcher_from(frames),
                    basis_id="test_basis", fetch_date="2026-09-04")
    manifest_path = bsp.manifest_path_for(str(cache_dir))
    text = (cache_dir / "B.csv").read_text(encoding="utf-8").splitlines()
    # Edit one close by a tiny amount: same row count, different bytes.
    last = text[-1].split(",")
    last[-1] = str(float(last[-1]) + 0.01)
    text[-1] = ",".join(last)
    (cache_dir / "B.csv").write_text("\n".join(text) + "\n", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        panel.load_cached_closes(
            str(cache_dir), ["A", "B", "C"], manifest_path=manifest_path)
    assert "B" in str(excinfo.value) and "sha256" in str(excinfo.value)


# --------------------------------------------------------------------------
# (c) the seal refuses a second write
# --------------------------------------------------------------------------

def test_sealed_cache_dir_refuses_a_second_write_cache(tmp_path):
    frames = _abc_frames(6)
    cache_dir = tmp_path / "cache"
    bsp.write_cache(str(cache_dir), list(frames), fetcher=_fetcher_from(frames),
                    basis_id="test_basis", fetch_date="2026-09-04")
    before = {name: (cache_dir / name).read_bytes()
              for name in os.listdir(cache_dir)}
    calls = []

    def counting_fetcher(ticker):
        calls.append(ticker)
        return _ohlc_frame(frames[ticker])

    with pytest.raises(ValueError) as excinfo:
        bsp.write_cache(str(cache_dir), list(frames), fetcher=counting_fetcher,
                        basis_id="test_basis", fetch_date="2026-09-05")
    assert "SEALED" in str(excinfo.value)
    assert calls == []  # refused before any fetch
    after = {name: (cache_dir / name).read_bytes()
             for name in os.listdir(cache_dir)}
    assert after == before  # not a byte changed


def test_unsealed_dir_with_leftover_csvs_is_refused(tmp_path):
    frames = _abc_frames(6)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "A.csv").write_text("Date,Close\n", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        bsp.write_cache(str(cache_dir), list(frames),
                        fetcher=_fetcher_from(frames), basis_id="test_basis")
    assert "unattested" in str(excinfo.value)


# --------------------------------------------------------------------------
# (d) the four indicators on a hand-countable three-member panel
# --------------------------------------------------------------------------

def test_build_sessions_is_the_sorted_unique_union_of_member_dates():
    sessions = _sessions(6)
    closes = {
        "A": _ramp(sessions, 100.0, 1.0, drop=(1, 2)),
        "B": _ramp(sessions[3:], 200.0, 1.0),
        "C": _ramp(sessions, 300.0, 1.0, drop=(0, 5)),
    }
    union = bsp.build_sessions(closes)
    assert union.equals(sessions)
    assert union.tz is None and union.is_monotonic_increasing
    assert not union.has_duplicates


def test_participation_on_two_up_one_down_is_two_thirds_after_seasoning(tmp_path):
    built = _build_abc_panel(tmp_path, n=12)
    series, params, seasoning, path_dependent = bsp.compute_indicator(
        built, "participation", {"window": 3, "min_members": 3})
    assert params == {"min_members": 3, "window": 3}
    assert seasoning == 3 and path_dependent is False
    # Sessions 0 and 1 are unseasoned (fewer than 3 own bars) and dropped.
    assert len(series) == 10
    assert series.index[0] == built.sessions[2]
    assert np.isfinite(series.to_numpy()).all()
    assert np.allclose(series.to_numpy(), 2.0 / 3.0)


def test_net_advance_ratio_defaults_min_members_and_refuses_an_empty_result(tmp_path):
    # min_members defaults to the pre-registered constant; the synthetic
    # panel has only three members, so the default must yield NO defined
    # session and a loud refusal rather than a quietly thinner denominator.
    built = _build_abc_panel(tmp_path, n=12)
    with pytest.raises(ValueError) as excinfo:
        bsp.compute_indicator(built, "net_advance_ratio", {})
    assert "no defined sessions" in str(excinfo.value)
    assert str(constants.PANEL_MIN_RESOLVED_MEMBERS) in str(excinfo.value)

    series, params, seasoning, path_dependent = bsp.compute_indicator(
        built, "net_advance_ratio", {"min_members": 3})
    assert params == {"min_members": 3}
    assert seasoning == 2 and path_dependent is False
    # Session 0 has no previous close and is dropped; (2 - 1) / 3 thereafter.
    assert len(series) == 11
    assert series.index[0] == built.sessions[1]
    assert np.allclose(series.to_numpy(), 1.0 / 3.0)


def test_oscillator_of_a_constant_ratio_is_exactly_zero(tmp_path):
    built = _build_abc_panel(tmp_path, n=12)
    series, params, seasoning, path_dependent = bsp.compute_indicator(
        built, "net_advance_oscillator",
        {"fast": 2, "slow": 4, "min_members": 3})
    assert params == {"fast": 2, "min_members": 3, "slow": 4}
    assert seasoning == 2 and path_dependent is True
    # EMA(constant) == constant for both spans, so fast - slow == 0 exactly.
    assert len(series) == 11
    assert np.isfinite(series.to_numpy()).all()
    assert (series.to_numpy() == 0.0).all()


def test_cumulative_ad_trend_of_a_constant_ratio_is_one_sixth(tmp_path):
    built = _build_abc_panel(tmp_path, n=12)
    series, params, seasoning, path_dependent = bsp.compute_indicator(
        built, "cumulative_ad_trend", {"trend_days": 2, "min_members": 3})
    assert params == {"min_members": 3, "trend_days": 2}
    assert seasoning == 2 and path_dependent is True
    # cumsum = k/3; trailing-2 mean = (2k-1)/6; difference = 1/6 for k >= 2.
    # k = 1 (session 1) has no trailing-2 mean and is dropped.
    assert len(series) == 10
    assert series.index[0] == built.sessions[2]
    assert np.isfinite(series.to_numpy()).all()
    assert np.allclose(series.to_numpy(), 1.0 / 6.0)


def test_compute_indicator_refuses_unknown_names_and_stray_parameters(tmp_path):
    built = _build_abc_panel(tmp_path, n=6)
    with pytest.raises(ValueError):
        bsp.compute_indicator(built, "advance_decline_line", {})
    with pytest.raises(ValueError) as excinfo:
        bsp.compute_indicator(built, "participation",
                              {"window": 3, "min_members": 3, "slow": 4})
    assert "slow" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        bsp.compute_indicator(built, "net_advance_oscillator",
                              {"fast": 2, "min_members": 3})
    assert "slow" in str(excinfo.value)
    with pytest.raises(ValueError):
        bsp.compute_indicator(built, "participation",
                              {"window": 3.0, "min_members": 3})


# --------------------------------------------------------------------------
# (e) snapshots round-trip and refuse a wrong sha
# --------------------------------------------------------------------------

def test_write_snapshot_loads_back_with_its_own_sha_and_refuses_a_wrong_one(tmp_path):
    built = _build_abc_panel(tmp_path, n=12)
    out = tmp_path / "snapshots" / "participation.json"
    report = bsp.write_snapshot(
        built, "participation", {"window": 3, "min_members": 3}, str(out),
        built_at=BUILT_AT)
    assert os.path.exists(out)
    assert report["n_sessions"] == 10
    assert report["first_date"] == "2020-01-03"
    assert report["broken_sessions"] == [] and report["omitted_sessions"] == []
    assert report["seasoning_window"] == 3 and report["path_dependent"] is False

    loaded = panel.load_breadth_snapshot(
        str(out), expect_sha256=report["content_sha256"])
    assert len(loaded) == 10
    assert loaded.as_of("2020-01-02") is None
    assert loaded.as_of("2020-01-03") == pytest.approx(2.0 / 3.0)
    assert loaded.as_of("2021-01-01") == pytest.approx(2.0 / 3.0)

    payload = json.loads(out.read_text(encoding="utf-8"))
    provenance = payload["provenance"]
    assert provenance["basis_id"] == "test_basis"
    assert provenance["adjustment"] == constants.PANEL_PRICE_ADJUSTMENT
    assert provenance["sessions_source"].startswith("union of member")
    assert provenance["membership_sha256"] == hashlib.sha256(
        (tmp_path / "membership.json").read_bytes()).hexdigest()
    assert provenance["built_at"] == "2026-09-04T21:00:00Z"
    assert provenance["point_in_time"] is True

    with pytest.raises(ValueError) as excinfo:
        panel.load_breadth_snapshot(str(out), expect_sha256="f" * 64)
    assert "pinned" in str(excinfo.value)


def test_write_snapshot_path_dependent_indicators_load_back(tmp_path):
    built = _build_abc_panel(tmp_path, n=12)
    out = tmp_path / "osc.json"
    report = bsp.write_snapshot(
        built, "net_advance_oscillator",
        {"fast": 2, "slow": 4, "min_members": 3}, str(out), built_at=BUILT_AT)
    assert report["path_dependent"] is True and report["seasoning_window"] == 2
    loaded = panel.load_breadth_snapshot(
        str(out), expect_sha256=report["content_sha256"])
    assert loaded.as_of("2020-01-31") == 0.0


def test_write_snapshot_requires_a_timezone_aware_built_at(tmp_path):
    built = _build_abc_panel(tmp_path, n=12)
    with pytest.raises(ValueError):
        bsp.write_snapshot(built, "participation",
                           {"window": 3, "min_members": 3},
                           str(tmp_path / "x.json"),
                           built_at=datetime(2026, 9, 4, 21, 0, 0))
    assert not os.path.exists(tmp_path / "x.json")


def test_parse_built_at_requires_an_offset():
    assert bsp.parse_built_at("2026-09-04T21:00:00Z") == BUILT_AT
    assert bsp.parse_built_at("2026-09-04T17:00:00-04:00") == BUILT_AT
    with pytest.raises(ValueError):
        bsp.parse_built_at("2026-09-04T21:00:00")
    with pytest.raises(ValueError):
        bsp.parse_built_at("yesterday")


# --------------------------------------------------------------------------
# (f) a fetch failure aborts and leaves nothing a loader would accept
# --------------------------------------------------------------------------

def test_a_fetch_failure_aborts_and_leaves_no_manifest_and_no_csvs(tmp_path):
    frames = _abc_frames(6)
    del frames["B"]  # B's fetch returns None, A has already been written
    cache_dir = tmp_path / "cache"
    with pytest.raises(RuntimeError) as excinfo:
        bsp.write_cache(str(cache_dir), ["A", "B", "C"],
                        fetcher=_fetcher_from(frames), basis_id="test_basis")
    message = str(excinfo.value)
    assert "B" in message and "ABORTED" in message
    assert not os.path.exists(bsp.manifest_path_for(str(cache_dir)))
    assert [name for name in os.listdir(cache_dir)
            if name.lower().endswith(".csv")] == []
    with pytest.raises(ValueError):
        panel.load_cached_closes(
            str(cache_dir), ["A"],
            manifest_path=bsp.manifest_path_for(str(cache_dir)))


def test_a_fetcher_exception_is_also_an_abort(tmp_path):
    frames = _abc_frames(6)

    def flaky(ticker):
        if ticker == "C":
            raise ConnectionError("simulated transport failure")
        return _ohlc_frame(frames[ticker])

    cache_dir = tmp_path / "cache"
    with pytest.raises(RuntimeError) as excinfo:
        bsp.write_cache(str(cache_dir), list(frames), fetcher=flaky,
                        basis_id="test_basis")
    assert "C" in str(excinfo.value)
    assert "simulated transport failure" in str(excinfo.value)
    assert not os.path.exists(bsp.manifest_path_for(str(cache_dir)))


def test_a_first_bar_later_than_the_recorded_start_aborts(tmp_path):
    frames = _abc_frames(6)
    # C's history begins two sessions after the membership file says it does.
    frames["C"] = frames["C"].iloc[2:]
    membership = _membership_abc()
    cache_dir = tmp_path / "cache"
    with pytest.raises(RuntimeError) as excinfo:
        bsp.write_cache(str(cache_dir), list(frames),
                        fetcher=_fetcher_from(frames), basis_id="test_basis",
                        membership=membership)
    assert "C" in str(excinfo.value) and "LATER" in str(excinfo.value)
    assert not os.path.exists(bsp.manifest_path_for(str(cache_dir)))

    # An EARLIER first bar is fine.
    frames = _abc_frames(6)
    membership = _membership_abc(start="2020-01-03")
    bsp.write_cache(str(cache_dir), list(frames),
                    fetcher=_fetcher_from(frames), basis_id="test_basis",
                    membership=membership)
    assert os.path.exists(bsp.manifest_path_for(str(cache_dir)))


def test_normalize_fetched_frame_strips_timezone_and_refuses_bad_closes():
    sessions = _sessions(4)
    closes = _ramp(sessions, 10.0, 1.0)
    aware = pd.DataFrame({
        "Open": closes.to_numpy(), "High": closes.to_numpy(),
        "Low": closes.to_numpy(), "Close": closes.to_numpy()},
        index=sessions.tz_localize("America/New_York"))
    out = bsp.normalize_fetched_frame("A", aware)
    assert out.index.tz is None and out.index.equals(sessions)
    assert list(out.columns) == list(bsp.CSV_COLUMNS)

    bad = _ohlc_frame(closes)
    bad.loc[2, "Close"] = 0.0
    with pytest.raises(ValueError):
        bsp.normalize_fetched_frame("A", bad)
    with pytest.raises(ValueError):
        bsp.normalize_fetched_frame("A", None)
    with pytest.raises(ValueError):
        bsp.normalize_fetched_frame("A", pd.DataFrame())
