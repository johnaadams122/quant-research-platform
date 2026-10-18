"""Tests for scripts/validate_alpaca_intraday.py's bisection logic.

This script is a real-API, not-part-of-pytest validation tool (see its own
module docstring) -- these tests cover only the pure bisection algorithm
(`_bisect_retention_boundary`) and the no-retry behaviour of
`_probe_has_data`, both of which are directly testable with zero real
network calls. They do NOT exercise `main()` or anything that touches the
live Alpaca API.

The property under test: a probe reports data presence for exactly the
window it probed. Substituting a different (for example a later, shifted)
window's result for the probed date would corrupt the bisection near the
true retention boundary -- a later date having data does not prove the
probed date does -- and would yield a falsely-early "data starts here"
conclusion. So `_probe_has_data` fetches only the probed window and
never retries on a shifted window (fetch_intraday_bars uses a real
NYSE calendar internally, so a holiday inside the probe week is no reason
to shift), and the bisection loop lives in `_bisect_retention_boundary`,
a small pure function taking a `has_data_fn(date) -> bool` callable, so
the algorithm's "never substitute a different date's result" property can
be verified directly rather than only by manual inspection.
"""

import os
import sys
from datetime import date

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _path in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import validate_alpaca_intraday as validate_script  # noqa: E402
from mining.intraday_data import IntradayDataGapError  # noqa: E402


# ---------------------------------------------------------------------------
# _bisect_retention_boundary -- pure algorithm, fake has_data_fn, zero I/O
# ---------------------------------------------------------------------------

def test_bisect_retention_boundary_converges_around_a_fake_boundary():
    boundary = date(2016, 3, 1)

    def has_data(probe_date):
        return probe_date >= boundary

    lo_bound, hi_bound, iterations = validate_script._bisect_retention_boundary(
        date(2005, 1, 1), date(2025, 1, 1), has_data,
    )

    assert lo_bound < boundary <= hi_bound
    assert (hi_bound - lo_bound).days <= 8
    assert iterations > 0


def test_bisect_retention_boundary_only_ever_queries_the_actual_midpoint():
    # Guards against querying date X but reporting a result attributed to
    # a DIFFERENT, later date. Recompute the expected midpoint sequence
    # independently and confirm every query the algorithm makes matches
    # it precisely -- no substituted dates anywhere.
    boundary = date(2016, 3, 1)
    queried = []

    def has_data(probe_date):
        queried.append(probe_date)
        return probe_date >= boundary

    lo, hi = date(2010, 1, 1), date(2020, 1, 1)
    lo_bound, hi_bound, iterations = validate_script._bisect_retention_boundary(lo, hi, has_data)

    expected = []
    exp_lo, exp_hi = lo, hi
    while (exp_hi - exp_lo).days > 8:
        mid = exp_lo + (exp_hi - exp_lo) // 2
        expected.append(mid)
        if mid >= boundary:
            exp_hi = mid
        else:
            exp_lo = mid

    assert queried == expected
    assert len(queried) == iterations
    assert (lo_bound, hi_bound) == (exp_lo, exp_hi)


def test_bisect_retention_boundary_on_probe_callback_receives_exact_values():
    boundary = date(2016, 3, 1)
    seen = []

    def has_data(probe_date):
        return probe_date >= boundary

    def on_probe(round_no, probe_date, result):
        seen.append((round_no, probe_date, result))

    validate_script._bisect_retention_boundary(
        date(2010, 1, 1), date(2020, 1, 1), has_data, on_probe,
    )

    assert seen
    assert all(result == (probe_date >= boundary) for _, probe_date, result in seen)
    assert [round_no for round_no, _, _ in seen] == list(range(1, len(seen) + 1))


def test_bisect_retention_boundary_converges_on_true_boundary_near_a_gap():
    # Exercises the shape a shifted-window substitution would get wrong,
    # against the real bisection function end-to-end: the true boundary
    # sits at a date such that some of the bisection's own midpoints fall
    # in the 1-27 days STRICTLY BEFORE that boundary (a real gap at that
    # exact probe) while a window 14-28 days later can already have data.
    # A correct bisection converges on `true_boundary` regardless; a probe
    # that substituted a window shifted 14-28 days later would report
    # "data present" up to 28 days too early whenever a probe landed in
    # that gap, systematically pulling the reported boundary earlier than
    # reality.
    true_boundary = date(2015, 12, 20)

    def has_data(probe_date):
        return probe_date >= true_boundary

    lo_bound, hi_bound, _ = validate_script._bisect_retention_boundary(
        date(2015, 10, 1), date(2016, 2, 1), has_data,
    )

    # The bracket must straddle the TRUE boundary exactly -- not a date
    # up to 28 days earlier, which is what a shifted-window substitution
    # would systematically produce whenever a probe fell in a gap just
    # before the real boundary.
    assert lo_bound < true_boundary <= hi_bound
    assert (hi_bound - lo_bound).days <= 8


# ---------------------------------------------------------------------------
# _probe_has_data -- probes only the requested window, no shifted-window retry
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload

    def json(self):
        return self._payload


class _AlwaysEmptySession:
    """Every request returns a real, valid, but EMPTY page (a genuine gap
    at whatever window is requested). Records every call made so the test
    can prove no retry-on-a-shifted-window ever happens.
    """

    def __init__(self):
        self.calls = 0

    def get(self, url, params=None, headers=None, timeout=None, allow_redirects=None):
        self.calls += 1
        return _FakeResponse({"bars": None, "next_page_token": None, "symbol": "QQQ"})


def test_probe_has_data_reports_absent_for_the_probed_date_with_no_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID", "fake-key-id")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "fake-secret")
    monkeypatch.setattr(validate_script, "DEFAULT_CACHE_DIR", str(tmp_path))

    session = _AlwaysEmptySession()
    probe_date = date(2020, 1, 6)  # an arbitrary Monday, no special significance

    has_data, start, end = validate_script._probe_has_data("QQQ", session, probe_date)

    assert has_data is False
    # Exactly ONE HTTP call. A shifted-window retry loop (for example
    # re-probing +14/+28 days later) would make up to 3 before giving up,
    # so this pins that no retry happens at all, not just that one is
    # unlikely to fire.
    assert session.calls == 1


def test_probe_has_data_propagates_gap_as_false_not_an_exception(tmp_path, monkeypatch):
    # fetch_intraday_bars raises IntradayDataGapError on a genuine gap;
    # _probe_has_data's whole job is to convert that into a plain
    # (False, start, end) result for the bisection to consume, not let it
    # propagate as an exception nor swallow it into a retry.
    monkeypatch.setenv("ALPACA_API_KEY_ID", "fake-key-id")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "fake-secret")
    monkeypatch.setattr(validate_script, "DEFAULT_CACHE_DIR", str(tmp_path))

    session = _AlwaysEmptySession()
    try:
        has_data, _, _ = validate_script._probe_has_data("QQQ", session, date(2020, 1, 6))
    except IntradayDataGapError:
        pytest.fail("_probe_has_data must not let IntradayDataGapError propagate")
    assert has_data is False
