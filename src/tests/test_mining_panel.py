"""Guards for the cross-sectional daily panel (src/mining/panel.py).

The load-bearing tests here are the seasoning rule and the
status-resolution ORDER. Both guard against one defect class: deriving
`seasoned` from a rolling statistic would let a fetch gap masquerade as a
real market fact, because the missing bar NaN-propagates into the rolling
mean and the member silently leaves the denominator instead of breaking the
date.

No network anywhere. Every price frame here is synthetic and built in the
test itself: no @pytest.fixture, module-level `_`-prefixed builders, tmp_path.
"""
import hashlib
import inspect
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import constants
from mining import panel


# --------------------------------------------------------------------------
# Builders. Synthetic sessions are plain business days -- a stand-in market
# calendar with no holidays, which keeps every expectation below hand-countable.
# --------------------------------------------------------------------------

def _sessions(n, start="2020-01-01"):
    return pd.DatetimeIndex(pd.bdate_range(start=start, periods=n))


def _membership(members, *, basis_id="test_basis", point_in_time=True,
                ledgerable=True, block_reason=None):
    """members: iterable of (ticker, start, end-or-None)."""
    return {
        "schema_version": 1,
        "basis_id": basis_id,
        "point_in_time": point_in_time,
        "ledgerable": ledgerable,
        "ledger_block_reason": block_reason,
        "members": [{"ticker": t, "start": s, "end": e} for t, s, e in members],
    }


def _ramp(sessions, first, step, *, drop=()):
    """A strictly monotone price series (up if step > 0, down if step < 0).

    A monotone-up member is above any trailing mean of its own history on
    every session the mean is defined; a monotone-down member is below it.
    That makes every participation fraction below an exact, hand-countable
    integer ratio, so no float tolerance is needed anywhere.
    """
    values = [float(first) + step * i for i in range(len(sessions))]
    series = pd.Series(values, index=pd.DatetimeIndex(sessions), dtype=float)
    if drop:
        series = series.drop(index=[sessions[i] for i in drop])
    return series


def _provenance(**overrides):
    base = {
        "basis_id": "test_basis",
        "membership_path": "data/mining/panel_membership_test.json",
        "membership_sha256": "0" * 64,
        "manifest_path": "data/mining/panel_snapshots/manifest_test.json",
        "panel_fingerprint": "1" * 64,
        "adjustment": constants.PANEL_PRICE_ADJUSTMENT,
        "sessions_source": "synthetic business days (unit test)",
    }
    base.update(overrides)
    return base


def _three_ticker_panel(n=10, *, drop_c=()):
    """A, B, C: all members from session 0. C optionally misses some bars."""
    sessions = _sessions(n)
    membership = _membership([
        ("A", "2020-01-01", None),
        ("B", "2020-01-01", None),
        ("C", "2020-01-01", None),
    ])
    prices = {
        "A": _ramp(sessions, 100.0, 1.0),
        "B": _ramp(sessions, 200.0, 1.0),
        # A gentle decay so a 1500-session panel never walks a price to zero;
        # the step's magnitude is ~5e-2, six orders of magnitude above any
        # float noise in the trailing-mean comparison, so "below its own mean"
        # stays unambiguous without a tolerance.
        "C": _ramp(sessions, 500.0, -0.05, drop=drop_c),
    }
    return panel.build_panel(
        prices_by_ticker=prices, membership=membership, sessions=sessions,
        provenance=_provenance())


# --------------------------------------------------------------------------
# The seasoning rule -- the most important guard in this module.
# --------------------------------------------------------------------------

def test_seasoning_is_derived_from_membership_not_from_a_rolling_statistic():
    # C is a seasoned member that simply has no bar on session 7. It must
    # BREAK that session, not quietly leave the denominator.
    built = _three_ticker_panel(n=10, drop_c=(7,))
    gap = built.sessions[7]

    seasoned = panel.seasoned_mask(built, window=3)
    assert bool(seasoned.loc[gap, "C"]) is True

    broken = panel.broken_sessions(built, window=3)
    assert bool(broken.loc[gap]) is True

    frac = panel.breadth_participation(built, window=3, min_members=1)
    assert math.isnan(frac.loc[gap])

    coverage = panel.coverage_report(built, window=3)
    # THREE seasoned members on that date, not two. Deriving `seasoned` from
    # prices.rolling(window).mean().notna() reports two, because the missing
    # bar NaN-propagates into the rolling mean and the member disappears.
    assert int(coverage.loc[gap, "seasoned_n"]) == 3
    assert int(coverage.loc[gap, "resolved_n"]) == 2
    assert bool(coverage.loc[gap, "broken"]) is True


def test_a_gap_breaks_only_its_own_session_not_the_window_that_follows_it():
    # The trailing mean is taken over each ticker's OWN bars, which is the only
    # definition coherent with the seasoning rule (cumulative own-bar count).
    # A frame-level prices.rolling(window).mean() would be NaN for window-1
    # sessions after ANY gap, blanking a healthy member for months over one
    # missing print -- an over-refusal, where a rolling-derived SEASONING
    # mask (see the test above) would be an under-refusal.
    built = _three_ticker_panel(n=20, drop_c=(5,))
    broken = panel.broken_sessions(built, window=3)
    assert bool(broken.loc[built.sessions[5]]) is True
    assert not broken.drop(index=[built.sessions[5]]).any()

    frac = panel.breadth_participation(built, window=3, min_members=1)
    assert math.isnan(frac.loc[built.sessions[5]])
    # A and B rising, C decaying: two of three members above their own mean.
    assert frac.loc[built.sessions[6]] == pytest.approx(2.0 / 3.0)
    assert frac.loc[built.sessions[8]] == pytest.approx(2.0 / 3.0)
    coverage = panel.coverage_report(built, window=3)
    assert int(coverage.loc[built.sessions[6], "resolved_n"]) == 3


def test_seasoning_counts_a_members_own_bars_not_calendar_sessions():
    # C joins late: its first bar is session 4, so with window=3 it cannot be
    # seasoned before session 6 no matter how many sessions have elapsed.
    sessions = _sessions(10)
    membership = _membership([
        ("A", "2020-01-01", None),
        ("C", str(sessions[4].date()), None),
    ])
    prices = {
        "A": _ramp(sessions, 100.0, 1.0),
        "C": _ramp(sessions[4:], 300.0, 1.0),
    }
    built = panel.build_panel(
        prices_by_ticker=prices, membership=membership, sessions=sessions,
        provenance=_provenance())

    seasoned = panel.seasoned_mask(built, window=3)
    assert bool(seasoned.loc[sessions[5], "C"]) is False
    assert bool(seasoned.loc[sessions[6], "C"]) is True
    # Before its membership start it is NOT_A_MEMBER, never UNSEASONED.
    status = panel.status_at_window(built, window=3)
    assert int(status.loc[sessions[0], "C"]) == int(panel.PanelStatus.NOT_A_MEMBER)
    assert int(status.loc[sessions[5], "C"]) == int(panel.PanelStatus.UNSEASONED)
    assert int(status.loc[sessions[6], "C"]) == int(panel.PanelStatus.OK)


# --------------------------------------------------------------------------
# A broken session is dropped and named, never approximated.
# --------------------------------------------------------------------------

def test_missing_bar_for_a_seasoned_member_makes_the_session_unknown_not_a_number(tmp_path):
    # A long panel so the single break stays under the pre-registered
    # broken-session fraction ceiling and the snapshot is allowed to be written.
    built = _three_ticker_panel(n=1500, drop_c=(800,))
    gap = built.sessions[800]

    frac = panel.breadth_participation(built, window=3, min_members=1)
    assert math.isnan(frac.loc[gap])

    out = tmp_path / "participation.json"
    result = panel.write_breadth_snapshot(
        frac.dropna(), str(out), indicator="test_participation_above_ma",
        panel=built, parameters={"window": 3}, seasoning_window=3,
        path_dependent=False,
        built_at=pd.Timestamp("2026-09-02T00:00:00Z").to_pydatetime())

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert str(gap.date()) not in payload["dates"]
    assert payload["provenance"]["broken_sessions"] == [str(gap.date())]
    assert result["content_sha256"] == payload["content_sha256"]
    # values never carries NaN -- an unknown session is absent, not estimated.
    assert all(math.isfinite(v) for v in payload["values"])


# --------------------------------------------------------------------------
# The status-resolution ORDER is the guarantee.
# --------------------------------------------------------------------------

def test_fetch_failure_is_never_reported_as_not_a_member_or_unseasoned():
    sessions = _sessions(10)
    membership = _membership([
        ("A", "2020-01-01", None),
        # D joins on session 4 AND its fetch failed. If the fetch-failure
        # check ran after the membership/seasoning checks, sessions 0-3 would
        # read NOT_A_MEMBER and 4-5 UNSEASONED -- i.e. we would be inferring a
        # listing life from bars we never received.
        ("D", str(sessions[4].date()), None),
    ])
    prices = {"A": _ramp(sessions, 100.0, 1.0)}

    status = panel.resolve_status(
        prices=pd.DataFrame(
            {"A": prices["A"], "D": pd.Series(index=sessions, dtype=float)},
            index=sessions),
        member=panel.membership_mask(membership, sessions),
        fetch_failed=frozenset({"D"}),
        quarantined=frozenset(),
        seasoned=pd.DataFrame(False, index=sessions, columns=["A", "D"]),
    )
    assert (status["D"] == int(panel.PanelStatus.FETCH_FAILED)).all()
    assert int(status.loc[sessions[0], "D"]) == int(panel.PanelStatus.FETCH_FAILED)
    assert int(status.loc[sessions[5], "D"]) == int(panel.PanelStatus.FETCH_FAILED)


def test_quarantine_outranks_membership_and_seasoning_but_not_fetch_failure():
    sessions = _sessions(6)
    membership = _membership([("A", "2020-01-01", None),
                              ("Q", str(sessions[3].date()), None)])
    status = panel.resolve_status(
        prices=pd.DataFrame(
            {"A": _ramp(sessions, 100.0, 1.0),
             "Q": pd.Series(index=sessions, dtype=float)}, index=sessions),
        member=panel.membership_mask(membership, sessions),
        fetch_failed=frozenset(),
        quarantined=frozenset({"Q"}),
        seasoned=pd.DataFrame(False, index=sessions, columns=["A", "Q"]),
    )
    assert (status["Q"] == int(panel.PanelStatus.QUARANTINED)).all()


def test_build_panel_raises_when_any_member_failed_to_fetch():
    sessions = _sessions(10)
    membership = _membership([("A", "2020-01-01", None),
                              ("D", "2020-01-01", None)])
    with pytest.raises(ValueError) as excinfo:
        panel.build_panel(
            prices_by_ticker={"A": _ramp(sessions, 100.0, 1.0)},
            membership=membership, sessions=sessions,
            fetch_failed=frozenset({"D"}), provenance=_provenance())
    message = str(excinfo.value)
    assert "D" in message
    assert str(constants.PANEL_MAX_FETCH_FAILURES) in message


# --------------------------------------------------------------------------
# A missing price is NaN, never 0.0.
# --------------------------------------------------------------------------

def test_a_missing_price_is_never_filled_with_zero():
    sessions = _sessions(10)
    membership = _membership([
        ("A", "2020-01-01", None),
        ("B", "2020-01-01", None),
        ("C", str(sessions[6].date()), None),
    ])
    prices = {
        "A": _ramp(sessions, 100.0, 1.0),
        "B": _ramp(sessions, 200.0, -1.0),
        "C": _ramp(sessions[6:], 300.0, 1.0),
    }
    built = panel.build_panel(
        prices_by_ticker=prices, membership=membership, sessions=sessions,
        provenance=_provenance())

    # A 0.00 close is a -100% return and an automatic "below its 200dma";
    # pooling.py's fill_value=0.0 union is correct for summing P&L and
    # catastrophic here.
    assert built.prices.loc[sessions[0], "C"] != 0.0
    assert bool(pd.isna(built.prices.loc[sessions[0], "C"]))
    assert bool((built.prices.fillna(-1.0) != 0.0).all().all())

    # C is not a member early, so it is neither an advancer nor a decliner:
    # the fraction is 1 of 2 (A up, B down), never 1 of 3.
    frac = panel.breadth_participation(built, window=3, min_members=1)
    assert frac.loc[sessions[4]] == 0.5


# --------------------------------------------------------------------------
# The min-resolved-members boundary is inclusive.
# --------------------------------------------------------------------------

def _n_member_panel(resolved_members, *, extra_late_member=False, n=20):
    sessions = _sessions(n)
    members = [(f"T{i}", "2020-01-01", None) for i in range(resolved_members)]
    prices = {f"T{i}": _ramp(sessions, 100.0 + i, 1.0 if i % 2 == 0 else -1.0)
              for i in range(resolved_members)}
    if extra_late_member:
        members.append(("LATE", str(sessions[15].date()), None))
        prices["LATE"] = _ramp(sessions[15:], 500.0, 1.0)
    return panel.build_panel(
        prices_by_ticker=prices, membership=_membership(members),
        sessions=sessions, provenance=_provenance())


def test_min_resolved_members_boundary_is_inclusive():
    exactly = constants.PANEL_MIN_RESOLVED_MEMBERS
    at_bar = _n_member_panel(exactly)
    below_bar = _n_member_panel(exactly - 1)
    probe = at_bar.sessions[10]

    at_value = panel.breadth_participation(
        at_bar, window=3, min_members=exactly).loc[probe]
    below_value = panel.breadth_participation(
        below_bar, window=3, min_members=exactly).loc[probe]

    # Exactly PANEL_MIN_RESOLVED_MEMBERS passes -- the house >=-passes
    # convention shared with DSR_PASS_THRESHOLD and MIN_TRADES_FOR_SCORING.
    assert not math.isnan(at_value)
    assert math.isnan(below_value)


# --------------------------------------------------------------------------
# Path-dependent indicators refuse to accumulate across a gap.
# --------------------------------------------------------------------------

def test_path_dependent_indicator_refuses_to_build_across_a_broken_session():
    built = _three_ticker_panel(n=60, drop_c=(30,))
    gap_label = str(built.sessions[30].date())

    # Pointwise: tolerated, dropped, NaN at the gap.
    pointwise = panel.breadth_participation(built, window=3, min_members=1)
    assert math.isnan(pointwise.loc[built.sessions[30]])

    with pytest.raises(ValueError) as osc:
        panel.breadth_net_advance_oscillator(
            built, fast=5, slow=9, min_members=1)
    assert gap_label in str(osc.value)

    with pytest.raises(ValueError) as cum:
        panel.breadth_cumulative_ad_trend(built, trend_days=5, min_members=1)
    assert gap_label in str(cum.value)

    # And prove the gap-fill would contaminate the WHOLE downstream path, not
    # one bar. Same ratio series, same cumsum, only difference is the gap:
    # dropping the unknown rows and accumulating (exactly what pandas does on
    # its own, since .cumsum() skips NaN) shifts EVERY subsequent value.
    clean = _three_ticker_panel(n=60)
    clean_ratio = panel.breadth_net_advance_ratio(clean, min_members=1)
    gapped_ratio = panel.breadth_net_advance_ratio(built, min_members=1)
    clean_cum = clean_ratio.dropna().cumsum()
    bridged_cum = gapped_ratio.dropna().cumsum()
    after_gap = bridged_cum.index[bridged_cum.index > built.sessions[31]]
    assert len(after_gap) > 10
    assert (clean_cum.reindex(after_gap).to_numpy()
            != bridged_cum.reindex(after_gap).to_numpy()).all()


def test_require_no_broken_sessions_names_every_broken_date():
    built = _three_ticker_panel(n=60, drop_c=(20, 41))
    first = str(built.sessions[20].date())
    second = str(built.sessions[41].date())

    with pytest.raises(ValueError) as excinfo:
        panel.require_no_broken_sessions(built, window=3)
    message = str(excinfo.value)
    assert first in message and second in message

    # A caller may bound the span it needs to be gap-free. .cumsum()/.ewm()
    # skip NaN, so "few broken sessions" is not a usable guarantee for a
    # path-dependent consumer -- "zero across THIS span" is.
    panel.require_no_broken_sessions(
        built, window=3, start=built.sessions[21], end=built.sessions[40])
    with pytest.raises(ValueError):
        panel.require_no_broken_sessions(
            built, window=3, start=built.sessions[21], end=built.sessions[41])


def test_net_advance_ratio_is_a_ratio_so_a_denominator_change_is_not_a_level_shift():
    # Two advancers and one decliner -> (2 - 1) / 3, not the raw count 1.
    sessions = _sessions(10)
    membership = _membership([("A", "2020-01-01", None),
                              ("B", "2020-01-01", None),
                              ("C", "2020-01-01", None)])
    prices = {
        "A": _ramp(sessions, 100.0, 1.0),
        "B": _ramp(sessions, 200.0, 1.0),
        "C": _ramp(sessions, 300.0, -1.0),
    }
    built = panel.build_panel(
        prices_by_ticker=prices, membership=membership, sessions=sessions,
        provenance=_provenance())
    nar = panel.breadth_net_advance_ratio(built, min_members=1)
    assert nar.loc[sessions[5]] == pytest.approx(1.0 / 3.0)


# --------------------------------------------------------------------------
# build_panel fails closed on structurally impossible inputs.
# --------------------------------------------------------------------------

def test_build_panel_raises_on_an_empty_membership():
    sessions = _sessions(5)
    with pytest.raises(ValueError):
        panel.build_panel(
            prices_by_ticker={}, membership=_membership([]), sessions=sessions,
            provenance=_provenance())


def test_build_panel_raises_when_sessions_do_not_cover_the_membership_span():
    sessions = _sessions(5, start="2020-06-01")
    membership = _membership([("A", "2019-01-02", None)])
    with pytest.raises(ValueError) as excinfo:
        panel.build_panel(
            prices_by_ticker={"A": _ramp(sessions, 100.0, 1.0)},
            membership=membership, sessions=sessions, provenance=_provenance())
    assert "A" in str(excinfo.value)


def test_build_panel_raises_when_a_member_has_no_price_series():
    sessions = _sessions(5)
    membership = _membership([("A", "2020-01-01", None),
                              ("B", "2020-01-01", None)])
    with pytest.raises(ValueError) as excinfo:
        panel.build_panel(
            prices_by_ticker={"A": _ramp(sessions, 100.0, 1.0)},
            membership=membership, sessions=sessions, provenance=_provenance())
    assert "B" in str(excinfo.value)


def test_build_panel_rejects_a_non_positive_or_non_finite_price():
    sessions = _sessions(5)
    membership = _membership([("A", "2020-01-01", None)])
    bad = _ramp(sessions, 100.0, 1.0)
    bad.iloc[2] = 0.0
    with pytest.raises(ValueError):
        panel.build_panel(
            prices_by_ticker={"A": bad}, membership=membership,
            sessions=sessions, provenance=_provenance())


def test_load_membership_rejects_empty_duplicate_or_undated_members(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_membership([("A", "2020-01-01", None)])),
                    encoding="utf-8")
    loaded = panel.load_membership(str(good))
    assert loaded["members"][0]["ticker"] == "A"

    for payload in (
            _membership([]),
            _membership([("A", "2020-01-01", None), ("A", "2020-01-02", None)]),
            _membership([("A", None, None)]),
            _membership([("A", "not-a-date", None)]),
    ):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            panel.load_membership(str(bad))


# --------------------------------------------------------------------------
# Cache integrity and the sealed-directory rule.
# --------------------------------------------------------------------------

def _write_cache(tmp_path, frames):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    per_ticker = {}
    for ticker, series in frames.items():
        csv_path = cache_dir / f"{ticker}.csv"
        frame = pd.DataFrame({"Close": series})
        frame.index.name = "Date"
        frame.to_csv(csv_path)
        raw = csv_path.read_bytes()
        per_ticker[ticker] = {
            "rows": int(len(series)),
            "first_bar": str(series.index[0].date()),
            "last_bar": str(series.index[-1].date()),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    manifest = {
        "schema_version": 1,
        "basis_id": "test_basis",
        "fetch_date": "2026-09-02",
        "adjustment": constants.PANEL_PRICE_ADJUSTMENT,
        "cache_dir": str(cache_dir),
        "membership_path": "data/mining/panel_membership_test.json",
        "membership_sha256": "0" * 64,
        "tickers_requested": sorted(frames),
        "tickers_cached": sorted(frames),
        "failed": {},
        "per_ticker": per_ticker,
        "panel_fingerprint": "1" * 64,
        "sealed_at": "2026-09-02T00:00:00Z",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return cache_dir, manifest_path, manifest


def test_cached_csv_sha256_mismatch_raises_naming_the_ticker(tmp_path):
    sessions = _sessions(6)
    cache_dir, manifest_path, _ = _write_cache(
        tmp_path, {"A": _ramp(sessions, 100.0, 1.0),
                   "B": _ramp(sessions, 200.0, 1.0)})

    loaded = panel.load_cached_closes(
        str(cache_dir), ["A", "B"], manifest_path=str(manifest_path))
    assert set(loaded) == {"A", "B"}
    assert len(loaded["A"]) == 6

    with open(cache_dir / "B.csv", "ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError) as excinfo:
        panel.load_cached_closes(
            str(cache_dir), ["A", "B"], manifest_path=str(manifest_path))
    assert "B" in str(excinfo.value)


def test_load_cached_closes_raises_on_a_ticker_absent_from_the_manifest(tmp_path):
    sessions = _sessions(6)
    cache_dir, manifest_path, _ = _write_cache(
        tmp_path, {"A": _ramp(sessions, 100.0, 1.0)})
    with pytest.raises(ValueError) as excinfo:
        panel.load_cached_closes(
            str(cache_dir), ["A", "MISSING"], manifest_path=str(manifest_path))
    assert "MISSING" in str(excinfo.value)


def test_load_cached_closes_raises_when_the_manifest_reports_a_failed_ticker(tmp_path):
    sessions = _sessions(6)
    cache_dir, manifest_path, manifest = _write_cache(
        tmp_path, {"A": _ramp(sessions, 100.0, 1.0)})
    manifest["failed"] = {"XLE": "yfinance returned no rows"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        panel.load_cached_closes(
            str(cache_dir), ["A"], manifest_path=str(manifest_path))
    assert "XLE" in str(excinfo.value)


def test_sealed_panel_cache_directory_refuses_topup(tmp_path):
    cache_dir = tmp_path / "test_basis_sealed_cache"
    cache_dir.mkdir()
    panel.assert_cache_dir_unsealed(str(cache_dir))  # empty dir is fine

    (cache_dir / panel.CACHE_SEAL_FILENAME).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        panel.assert_cache_dir_unsealed(str(cache_dir))
    assert panel.CACHE_SEAL_FILENAME in str(excinfo.value)


# --------------------------------------------------------------------------
# The snapshot is the evidence a ledger verdict rests on.
# --------------------------------------------------------------------------

def _write_snapshot(tmp_path, *, name="snap.json", point_in_time=True,
                    ledgerable=True, block_reason=None, extra_session=False):
    built = _three_ticker_panel(n=1500 + (1 if extra_session else 0))
    frac = panel.breadth_participation(built, window=3, min_members=1)
    out = tmp_path / name
    result = panel.write_breadth_snapshot(
        frac.dropna(), str(out), indicator="test_participation_above_ma",
        panel=built, parameters={"window": 3}, seasoning_window=3,
        path_dependent=False,
        built_at=pd.Timestamp("2026-09-02T00:00:00Z").to_pydatetime())
    if not (point_in_time and ledgerable):
        payload = json.loads(out.read_text(encoding="utf-8"))
        payload["provenance"]["point_in_time"] = point_in_time
        payload["provenance"]["ledgerable"] = ledgerable
        payload["provenance"]["ledger_block_reason"] = block_reason
        out.write_text(json.dumps(payload), encoding="utf-8")
    return out, result["content_sha256"]


def test_load_breadth_snapshot_round_trips_a_point_in_time_snapshot(tmp_path):
    out, sha = _write_snapshot(tmp_path)
    series = panel.load_breadth_snapshot(str(out), expect_sha256=sha)
    assert isinstance(series, panel.AsOfSeries)
    assert series.content_sha256 == sha
    assert series.indicator == "test_participation_above_ma"
    assert len(series) > 1000


def test_load_breadth_snapshot_raises_on_a_non_point_in_time_basis(tmp_path):
    out, sha = _write_snapshot(
        tmp_path, point_in_time=False, ledgerable=False,
        block_reason="membership has no vintage dimension")
    with pytest.raises(ValueError) as excinfo:
        panel.load_breadth_snapshot(str(out), expect_sha256=sha)
    message = str(excinfo.value)
    assert "test_basis" in message
    assert "membership has no vintage dimension" in message


def test_load_breadth_snapshot_has_no_override_parameter():
    signature = inspect.signature(panel.load_breadth_snapshot)
    assert not any("allow" in name for name in signature.parameters)
    assert not any("force" in name for name in signature.parameters)
    expect = signature.parameters["expect_sha256"]
    assert expect.kind is inspect.Parameter.KEYWORD_ONLY
    assert expect.default is inspect.Parameter.empty


def test_load_breadth_snapshot_raises_on_a_content_sha256_mismatch(tmp_path):
    old_out, old_sha = _write_snapshot(tmp_path, name="old.json")
    new_out, new_sha = _write_snapshot(
        tmp_path, name="new.json", extra_session=True)
    assert old_sha != new_sha

    with pytest.raises(ValueError) as excinfo:
        panel.load_breadth_snapshot(str(new_out), expect_sha256=old_sha)
    message = str(excinfo.value)
    assert old_sha in message
    assert new_sha in message


def test_load_breadth_snapshot_raises_when_the_stored_sha_does_not_match_content(tmp_path):
    out, sha = _write_snapshot(tmp_path)
    payload = json.loads(out.read_text(encoding="utf-8"))
    payload["values"][10] = payload["values"][10] + 0.25
    out.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        panel.load_breadth_snapshot(str(out), expect_sha256=sha)


def test_write_breadth_snapshot_refuses_a_non_point_in_time_basis(tmp_path):
    sessions = _sessions(40)
    membership = _membership(
        [("A", "2020-01-01", None), ("B", "2020-01-01", None)],
        point_in_time=False, ledgerable=False,
        block_reason="today's index membership has no vintage dimension")
    built = panel.build_panel(
        prices_by_ticker={"A": _ramp(sessions, 100.0, 1.0),
                          "B": _ramp(sessions, 200.0, -1.0)},
        membership=membership, sessions=sessions, provenance=_provenance())
    frac = panel.breadth_participation(built, window=3, min_members=1)
    with pytest.raises(ValueError) as excinfo:
        panel.write_breadth_snapshot(
            frac.dropna(), str(tmp_path / "s.json"), indicator="x",
            panel=built, parameters={"window": 3}, seasoning_window=3,
            path_dependent=False,
            built_at=pd.Timestamp("2026-09-02T00:00:00Z").to_pydatetime())
    assert "vintage" in str(excinfo.value)
    assert not (tmp_path / "s.json").exists()


def test_write_breadth_snapshot_refuses_when_broken_fraction_exceeds_the_ceiling(tmp_path):
    # 500 sessions with one break is 1/498 = 0.00201, above the
    # pre-registered PANEL_MAX_BROKEN_SESSION_FRACTION of 0.001.
    built = _three_ticker_panel(n=500, drop_c=(250,))
    frac = panel.breadth_participation(built, window=3, min_members=1)
    with pytest.raises(ValueError) as excinfo:
        panel.write_breadth_snapshot(
            frac.dropna(), str(tmp_path / "s.json"), indicator="x",
            panel=built, parameters={"window": 3}, seasoning_window=3,
            path_dependent=False,
            built_at=pd.Timestamp("2026-09-02T00:00:00Z").to_pydatetime())
    assert str(constants.PANEL_MAX_BROKEN_SESSION_FRACTION) in str(excinfo.value)
    assert not (tmp_path / "s.json").exists()


def test_write_breadth_snapshot_rejects_a_series_containing_nan(tmp_path):
    built = _three_ticker_panel(n=1500)
    frac = panel.breadth_participation(built, window=3, min_members=1)
    with pytest.raises(ValueError):
        panel.write_breadth_snapshot(
            frac, str(tmp_path / "s.json"), indicator="x", panel=built,
            parameters={"window": 3}, seasoning_window=3, path_dependent=False,
            built_at=pd.Timestamp("2026-09-02T00:00:00Z").to_pydatetime())


def test_write_breadth_snapshot_records_denominator_eras_and_provenance(tmp_path):
    built = _three_ticker_panel(n=1500)
    frac = panel.breadth_participation(built, window=3, min_members=1)
    out = tmp_path / "s.json"
    panel.write_breadth_snapshot(
        frac.dropna(), str(out), indicator="x", panel=built,
        parameters={"window": 3}, seasoning_window=3, path_dependent=False,
        built_at=pd.Timestamp("2026-09-02T00:00:00Z").to_pydatetime())
    provenance = json.loads(out.read_text(encoding="utf-8"))["provenance"]
    assert provenance["denominator_by_size"] == {"3": len(frac.dropna())}
    assert provenance["point_in_time"] is True
    assert provenance["ledgerable"] is True
    assert provenance["broken_sessions"] == []
    assert provenance["path_dependent"] is False
    assert "cumulative" in provenance["seasoning_rule"]
    assert provenance["built_at"] == "2026-09-02T00:00:00Z"


# --------------------------------------------------------------------------
# A session ABSENT from the series is an unknown session too.
#
# Validating only the converse (every series date must be a panel session) and
# measuring holes with coverage_report's narrow `broken` column (a seasoned
# member with no print) is not enough: a session dropped for any OTHER reason
# -- a min_members shortfall, an indicator's own lookback -- would be invisible
# to the ceiling AND to provenance, and AsOfSeries.as_of forward-fills across
# such a hole without bound: a membership change can leave hundreds of
# consecutive absent sessions with zero broken sessions.
# --------------------------------------------------------------------------

def _membership_change_panel(*, n, leave_at, join_at):
    """A and B throughout; X leaves after ``leave_at``; Y joins at ``join_at``.

    An ordinary index event. With min_members=3, every session from X's
    departure until Y is seasoned has only
    two resolved members, so breadth_participation is NaN there and the caller's
    .dropna() erases those sessions leaving no trace in the series itself.
    """
    sessions = _sessions(n)
    membership = _membership([
        ("A", "2020-01-01", None),
        ("B", "2020-01-01", None),
        ("X", "2020-01-01", str(sessions[leave_at].date())),
        ("Y", str(sessions[join_at].date()), None),
    ])
    prices = {
        "A": _ramp(sessions, 100.0, 1.0),
        "B": _ramp(sessions, 200.0, 1.0),
        "X": _ramp(sessions[:leave_at + 1], 300.0, 1.0),
        "Y": _ramp(sessions[join_at:], 400.0, 1.0),
    }
    return panel.build_panel(
        prices_by_ticker=prices, membership=membership, sessions=sessions,
        provenance=_provenance())


def test_sessions_absent_from_the_series_are_named_in_provenance(tmp_path):
    # 3000 sessions with a two-session hole: 2/2998 = 0.00067, under the
    # pre-registered ceiling, so the write is allowed -- and must still SAY so.
    built = _membership_change_panel(n=3000, leave_at=800, join_at=801)
    frac = panel.breadth_participation(built, window=3, min_members=3)
    hole = [str(built.sessions[801].date()), str(built.sessions[802].date())]
    assert all(math.isnan(frac.loc[built.sessions[i]]) for i in (801, 802))

    out = tmp_path / "s.json"
    panel.write_breadth_snapshot(
        frac.dropna(), str(out), indicator="x", panel=built,
        parameters={"window": 3}, seasoning_window=3, path_dependent=False,
        built_at=pd.Timestamp("2026-09-02T00:00:00Z").to_pydatetime())
    provenance = json.loads(out.read_text(encoding="utf-8"))["provenance"]

    # No member is missing a print anywhere, so the NARROW broken list is empty;
    # the omitted-session list is what records this hole.
    assert provenance["broken_sessions"] == []
    assert provenance["omitted_sessions"] == hole
    assert provenance["unknown_session_fraction"] == pytest.approx(2.0 / 2998.0)


def test_write_breadth_snapshot_refuses_a_long_run_of_absent_sessions(tmp_path):
    # 900 sessions, X leaves at 300, Y joins at 600, min_members=3 -> 301
    # consecutive absent sessions and zero broken sessions under the narrow
    # definition: the write must still be refused.
    built = _membership_change_panel(n=900, leave_at=300, join_at=600)
    frac = panel.breadth_participation(built, window=3, min_members=3)
    dropped = frac.dropna()
    assert len(dropped) == 597

    out = tmp_path / "s.json"
    with pytest.raises(ValueError) as excinfo:
        panel.write_breadth_snapshot(
            dropped, str(out), indicator="x", panel=built,
            parameters={"window": 3}, seasoning_window=3, path_dependent=False,
            built_at=pd.Timestamp("2026-09-02T00:00:00Z").to_pydatetime())
    message = str(excinfo.value)
    assert str(constants.PANEL_MAX_BROKEN_SESSION_FRACTION) in message
    assert str(built.sessions[301].date()) in message
    assert not out.exists()


def test_path_dependent_snapshot_refuses_an_absent_session_not_only_a_broken_one(tmp_path):
    # path_dependent is caller-declared, so this hands the same under-ceiling
    # series to the path-dependent branch. Zero broken sessions, two absent
    # ones: .cumsum()/.ewm() would bridge those exactly as they bridge a broken
    # one, so the branch must refuse on the union rather than on broken alone.
    built = _membership_change_panel(n=3000, leave_at=800, join_at=801)
    frac = panel.breadth_participation(built, window=3, min_members=3)
    out = tmp_path / "s.json"
    with pytest.raises(ValueError) as excinfo:
        panel.write_breadth_snapshot(
            frac.dropna(), str(out), indicator="x", panel=built,
            parameters={"window": 3}, seasoning_window=3, path_dependent=True,
            built_at=pd.Timestamp("2026-09-02T00:00:00Z").to_pydatetime())
    message = str(excinfo.value)
    assert "path-dependent" in message
    assert str(built.sessions[801].date()) in message
    assert not out.exists()


# --------------------------------------------------------------------------
# The content hash covers every claim the file makes, not just numbers.
# --------------------------------------------------------------------------

def test_content_sha256_covers_provenance_not_only_the_numbers(tmp_path):
    built = _three_ticker_panel(n=1500)
    frac = panel.breadth_participation(built, window=3, min_members=1)
    out = tmp_path / "s.json"
    sha = panel.write_breadth_snapshot(
        frac.dropna(), str(out), indicator="x", panel=built,
        parameters={"window": 3}, seasoning_window=3, path_dependent=False,
        built_at=pd.Timestamp("2026-09-02T00:00:00Z").to_pydatetime()
    )["content_sha256"]
    panel.load_breadth_snapshot(str(out), expect_sha256=sha)  # baseline: loads

    # Not one date and not one value is touched. Only the record of HOW the
    # numbers were produced -- including the broken-session list that is the
    # only durable evidence of data holes.
    payload = json.loads(out.read_text(encoding="utf-8"))
    payload["provenance"]["broken_sessions"] = ["FORGED"]
    payload["provenance"]["seasoning_rule"] = "whatever I like"
    payload["provenance"]["denominator_by_size"] = {"999": 1}
    payload["provenance"]["built_at"] = "1999-01-01T00:00:00Z"
    out.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        panel.load_breadth_snapshot(str(out), expect_sha256=sha)
    assert "sha256" in str(excinfo.value)


def test_content_sha256_covers_the_schema_version(tmp_path):
    built = _three_ticker_panel(n=1500)
    frac = panel.breadth_participation(built, window=3, min_members=1)
    out = tmp_path / "s.json"
    sha = panel.write_breadth_snapshot(
        frac.dropna(), str(out), indicator="x", panel=built,
        parameters={"window": 3}, seasoning_window=3, path_dependent=False,
        built_at=pd.Timestamp("2026-09-02T00:00:00Z").to_pydatetime()
    )["content_sha256"]

    payload = json.loads(out.read_text(encoding="utf-8"))
    payload["schema_version"] = panel.SNAPSHOT_SCHEMA_VERSION + 1
    out.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        panel.load_breadth_snapshot(str(out), expect_sha256=sha)
    assert "schema_version" in str(excinfo.value)


# --------------------------------------------------------------------------
# A snapshot that cannot be serialized is refused, never half-written.
# --------------------------------------------------------------------------

def _two_ticker_panel(*, provenance):
    sessions = _sessions(40)
    return panel.build_panel(
        prices_by_ticker={"A": _ramp(sessions, 100.0, 1.0),
                          "B": _ramp(sessions, 200.0, 1.0)},
        membership=_membership([("A", "2020-01-01", None),
                                ("B", "2020-01-01", None)]),
        sessions=sessions, provenance=provenance)


def test_write_refuses_unserializable_provenance_by_name_and_writes_nothing(tmp_path):
    # A provenance dict about a fetch carrying a pd.Timestamp is an entirely
    # ordinary thing. A TypeError from inside json.dump, after open(path, 'w')
    # had truncated the destination, would leave a partial JSON file a ledger
    # entry might already point at -- so it is refused by name first.
    built = _two_ticker_panel(
        provenance=_provenance(fetched_at=pd.Timestamp("2026-09-02")))
    frac = panel.breadth_participation(built, window=3, min_members=1)
    out = tmp_path / "s.json"
    with pytest.raises(ValueError) as excinfo:
        panel.write_breadth_snapshot(
            frac.dropna(), str(out), indicator="x", panel=built,
            parameters={"window": 3}, seasoning_window=3, path_dependent=False,
            built_at=pd.Timestamp("2026-09-02T00:00:00Z").to_pydatetime())
    assert "fetched_at" in str(excinfo.value)
    # Not the file, and not a temp file beside it either.
    assert list(tmp_path.iterdir()) == []


def test_write_refuses_unserializable_parameters_by_name(tmp_path):
    built = _two_ticker_panel(provenance=_provenance())
    frac = panel.breadth_participation(built, window=3, min_members=1)
    out = tmp_path / "s.json"
    with pytest.raises(ValueError) as excinfo:
        panel.write_breadth_snapshot(
            frac.dropna(), str(out), indicator="x", panel=built,
            parameters={"window": np.int64(3)}, seasoning_window=3,
            path_dependent=False,
            built_at=pd.Timestamp("2026-09-02T00:00:00Z").to_pydatetime())
    assert "window" in str(excinfo.value)
    assert list(tmp_path.iterdir()) == []


def test_a_failed_write_leaves_the_previous_snapshot_and_no_litter(tmp_path,
                                                                   monkeypatch):
    """The destination holds the OLD bytes or the complete new ones, never both.

    Pre-validation catches the serialization failures we know about; this proves
    the write itself is atomic for the ones we do not, by failing the rename.
    """
    built = _three_ticker_panel(n=1500)
    frac = panel.breadth_participation(built, window=3, min_members=1)
    out = tmp_path / "s.json"
    panel.write_breadth_snapshot(
        frac.dropna(), str(out), indicator="x", panel=built,
        parameters={"window": 3}, seasoning_window=3, path_dependent=False,
        built_at=pd.Timestamp("2026-09-02T00:00:00Z").to_pydatetime())
    original = out.read_bytes()

    def _rename_fails(src, dst):
        raise OSError("simulated: the rename failed")

    monkeypatch.setattr(panel.os, "replace", _rename_fails)
    with pytest.raises(OSError):
        panel.write_breadth_snapshot(
            frac.dropna(), str(out), indicator="different_indicator",
            panel=built, parameters={"window": 3}, seasoning_window=3,
            path_dependent=False,
            built_at=pd.Timestamp("2026-09-03T00:00:00Z").to_pydatetime())

    assert out.read_bytes() == original
    assert [entry.name for entry in tmp_path.iterdir()] == ["s.json"]


# --------------------------------------------------------------------------
# AsOfSeries is the structural look-ahead guard on the panel axis.
# --------------------------------------------------------------------------

def _as_of_series():
    dates = pd.DatetimeIndex(["2020-01-02", "2020-01-06", "2020-01-09"])
    # Exactly representable binary fractions: no tolerance needed anywhere.
    return panel.AsOfSeries(
        dates=dates, values=[0.25, 0.5, 0.75],
        indicator="test_participation_above_ma", content_sha256="a" * 64)


def test_as_of_series_exposes_no_index_slice_or_iteration_surface():
    series = _as_of_series()
    for forbidden in ("loc", "iloc", "index", "values", "__getitem__",
                      "__iter__", "reindex", "shift"):
        assert not hasattr(series, forbidden), forbidden
    assert set(type(series).__slots__) == {
        "_dates", "_values", "_indicator", "_content_sha256"}


def test_as_of_never_returns_a_value_dated_after_the_query():
    series = _as_of_series()
    t0 = pd.Timestamp("2020-01-02")
    t1 = pd.Timestamp("2020-01-06")
    t2 = pd.Timestamp("2020-01-09")

    assert series.as_of(t1) == 0.5
    assert series.as_of(t1 - pd.Timedelta(1, "ns")) == 0.25
    assert series.as_of(t0) == 0.25
    assert series.as_of(t0 - pd.Timedelta(days=1)) is None
    assert series.as_of(t2 + pd.Timedelta(days=30)) == 0.75


def test_as_of_lookup_is_bounded_per_call_regardless_of_series_length(monkeypatch):
    dates = pd.DatetimeIndex(pd.bdate_range("2000-01-03", periods=5000))
    series = panel.AsOfSeries(
        dates=dates, values=[0.5] * 5000, indicator="x",
        content_sha256="b" * 64)

    calls = {"n": 0}
    real = panel.np.searchsorted

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(panel.np, "searchsorted", counting)
    for i in range(500):
        series.as_of(dates[i])
    # One bounded O(log n) lookup per query -- the structural defense against
    # recomputing over the whole frame on every query, which is quadratic.
    assert calls["n"] == 500


def test_as_of_series_rejects_unsorted_or_non_finite_input():
    with pytest.raises(ValueError):
        panel.AsOfSeries(
            dates=pd.DatetimeIndex(["2020-01-06", "2020-01-02"]),
            values=[0.25, 0.5], indicator="x", content_sha256="c" * 64)
    with pytest.raises(ValueError):
        panel.AsOfSeries(
            dates=pd.DatetimeIndex(["2020-01-02", "2020-01-06"]),
            values=[0.25, float("nan")], indicator="x",
            content_sha256="c" * 64)
    with pytest.raises(ValueError):
        panel.AsOfSeries(
            dates=pd.DatetimeIndex([]), values=[], indicator="x",
            content_sha256="c" * 64)


def test_as_of_rejects_a_timezone_aware_query():
    series = _as_of_series()
    with pytest.raises(ValueError):
        series.as_of(pd.Timestamp("2020-01-06", tz="UTC"))


# --------------------------------------------------------------------------
# Calendar + coverage reporting.
# --------------------------------------------------------------------------

def test_sessions_between_returns_tz_naive_sorted_unique_nyse_sessions():
    sessions = panel.sessions_between("2024-07-01", "2024-07-08")
    assert [str(d.date()) for d in sessions] == [
        "2024-07-01", "2024-07-02", "2024-07-03", "2024-07-05", "2024-07-08"]
    assert sessions.tz is None
    assert sessions.is_monotonic_increasing
    assert not sessions.has_duplicates


def test_sessions_between_raises_on_an_inverted_or_empty_span():
    with pytest.raises(ValueError):
        panel.sessions_between("2024-07-08", "2024-07-01")
    with pytest.raises(ValueError):
        # A weekend-only span contains no sessions at all.
        panel.sessions_between("2024-07-06", "2024-07-07")


def test_coverage_report_columns_and_expected_counts():
    built = _three_ticker_panel(n=10, drop_c=(7,))
    coverage = panel.coverage_report(built, window=3)
    assert list(coverage.columns) == [
        "expected_n", "resolved_n", "seasoned_n", "broken"]
    early = built.sessions[0]
    assert int(coverage.loc[early, "expected_n"]) == 3
    assert int(coverage.loc[early, "seasoned_n"]) == 0
    assert bool(coverage.loc[early, "broken"]) is False
