import os
import pathlib
import sys

import pandas as pd
import pytest
import requests

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import intraday_data
from mining.intraday_data import (
    AlpacaRequestError,
    CredentialError,
    IntradayDataGapError,
    PaginationError,
    fetch_intraday_bars,
)


# ---------------------------------------------------------------------------
# Small local fakes -- zero real HTTP calls anywhere in this file. Neither
# fake ever constructs or references a real Alpaca URL; the FakeSession
# only records whatever the module under test passes it and returns canned
# payloads from an in-memory queue.
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload=None, status_code=200):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


class _FakeSession:
    """Returns queued canned responses in order; records every call made."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None, allow_redirects=None):
        self.calls.append({
            "url": url,
            "params": dict(params or {}),
            "headers": dict(headers or {}),
            "timeout": timeout,
            "allow_redirects": allow_redirects,
        })
        if not self._responses:
            raise AssertionError("FakeSession: no more canned responses queued")
        response = self._responses.pop(0)
        return response if isinstance(response, _FakeResponse) else _FakeResponse(response)


class _NoCallSession:
    """A session stub that fails the test if .get() is ever invoked."""

    def get(self, *args, **kwargs):
        raise AssertionError("HTTP call made when none should have been necessary")


_REAL_CREDENTIALS = lambda: ("fake-key-id", "fake-secret")  # noqa: E731
_NO_CREDENTIALS = lambda: (None, None)  # noqa: E731


def _bar(ts_iso, o, h, l, c, v, n=1, vw=None):
    return {"t": ts_iso, "o": o, "h": h, "l": l, "c": c, "v": v, "n": n,
            "vw": vw if vw is not None else c}


def _page(bars, next_page_token=None, symbol="QQQ"):
    return {"bars": bars, "symbol": symbol, "next_page_token": next_page_token}


# ---------------------------------------------------------------------------
# RTH filtering
# ---------------------------------------------------------------------------

def test_rth_filter_is_half_open_930_inclusive_1600_exclusive(tmp_path):
    # 2026-01-02 is a Friday in EST (UTC-5): 09:30 ET == 14:30Z, 16:00 ET == 21:00Z.
    #
    # Alpaca bar timestamps mark the START of their minute interval (a real
    # API-contract fact): a bar timestamped 16:00:00 ET covers 16:00:00-16:00:59 --
    # AFTER the 16:00:00 close, not the last minute of RTH. The window is
    # therefore half-open: [9:30:00, 16:00:00) -- 9:30 inclusive (the
    # first RTH bar), 16:00 EXCLUDED, and the last real RTH bar is
    # 15:59:00 (covering 15:59:00-15:59:59, ending exactly at the close).
    bars = [
        _bar("2026-01-02T14:29:00Z", 1, 1, 1, 1, 100),   # 09:29 ET -- excluded (before open)
        _bar("2026-01-02T14:30:00Z", 2, 2, 2, 2, 200),   # 09:30 ET -- included (first RTH bar)
        _bar("2026-01-02T20:59:00Z", 3, 3, 3, 3, 300),   # 15:59 ET -- included (last RTH bar)
        _bar("2026-01-02T21:00:00Z", 4, 4, 4, 4, 400),   # 16:00 ET -- excluded (covers the post-close minute)
        _bar("2026-01-02T21:01:00Z", 5, 5, 5, 5, 500),   # 16:01 ET -- excluded
    ]
    session = _FakeSession([_page(bars)])

    df = fetch_intraday_bars(
        "QQQ", "2026-01-02", "2026-01-02",
        cache_dir=str(tmp_path), session=session, credentials_fn=_REAL_CREDENTIALS,
    )

    assert len(df) == 2
    times = sorted(ts.time().isoformat() for ts in df.index)
    assert times == ["09:30:00", "15:59:00"]
    first = df[df.index.map(lambda t: t.time().isoformat() == "09:30:00")]
    last = df[df.index.map(lambda t: t.time().isoformat() == "15:59:00")]
    assert first["Close"].iloc[0] == 2
    assert last["Close"].iloc[0] == 3


def test_real_nyse_early_close_day_uses_actual_close_not_1600(tmp_path):
    # Guards against an RTH filter that hardcodes a 16:00 ET close for
    # every session: on a real NYSE half day the market closes at 13:00
    # ET, but Alpaca's feed can still report activity in the 13:00-16:00
    # window that is NOT part of that day's regular session. 2023-11-24
    # (the Friday after Thanksgiving) is a real scheduled early close, as
    # this call shows:
    # pandas_market_calendars.get_calendar("NYSE").schedule(
    #     start_date="2023-11-24", end_date="2023-11-24"
    # )["market_close"] returns 2023-11-24 18:00:00+00:00, i.e. 13:00:00
    # ET (a normal day's close on the same calendar is 21:00 UTC, 16:00
    # ET -- see test_real_nyse_holiday_inside_range_does_not_raise_gap_error
    # for the surrounding week's normal-day closes).
    bars = [
        _bar("2023-11-24T14:30:00Z", 1, 1, 1, 1, 100),  # 09:30 ET -- kept
        _bar("2023-11-24T17:59:00Z", 2, 2, 2, 2, 200),  # 12:59 ET -- kept (last half-day minute)
        _bar("2023-11-24T18:00:00Z", 3, 3, 3, 3, 300),  # 13:00 ET -- dropped (real close, half-open bound)
        _bar("2023-11-24T19:00:00Z", 4, 4, 4, 4, 400),  # 14:00 ET -- dropped (would be RTH on a normal day)
    ]
    session = _FakeSession([_page(bars)])

    df = fetch_intraday_bars(
        "QQQ", "2023-11-24", "2023-11-24",
        cache_dir=str(tmp_path), session=session, credentials_fn=_REAL_CREDENTIALS,
    )

    assert len(df) == 2
    times = sorted(ts.time().isoformat() for ts in df.index)
    assert times == ["09:30:00", "12:59:00"]


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------

def test_cache_miss_writes_a_file_readable_by_a_subsequent_call(tmp_path):
    bars = [_bar("2026-01-02T14:30:00Z", 10, 11, 9, 10.5, 500)]
    session = _FakeSession([_page(bars)])

    first = fetch_intraday_bars(
        "QQQ", "2026-01-02", "2026-01-02",
        cache_dir=str(tmp_path), session=session, credentials_fn=_REAL_CREDENTIALS,
    )
    # Cache key is (symbol, adjustment, day) -- see
    # test_different_adjustment_values_do_not_share_a_cache_entry.
    expected_path = os.path.join(str(tmp_path), "QQQ", "split", "2026-01-02.csv")
    assert os.path.exists(expected_path)

    second = fetch_intraday_bars(
        "QQQ", "2026-01-02", "2026-01-02",
        cache_dir=str(tmp_path), session=_NoCallSession(), credentials_fn=_REAL_CREDENTIALS,
    )
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


def test_cache_hit_avoids_any_http_call(tmp_path):
    bars = [_bar("2026-01-02T14:30:00Z", 10, 11, 9, 10.5, 500)]
    warm_session = _FakeSession([_page(bars)])
    fetch_intraday_bars(
        "QQQ", "2026-01-02", "2026-01-02",
        cache_dir=str(tmp_path), session=warm_session, credentials_fn=_REAL_CREDENTIALS,
    )
    assert len(warm_session.calls) == 1

    # Second call over the SAME fully-cached range must make zero HTTP
    # calls -- proven by a session stub that raises if .get() is invoked at
    # all, not just counted.
    df = fetch_intraday_bars(
        "QQQ", "2026-01-02", "2026-01-02",
        cache_dir=str(tmp_path), session=_NoCallSession(), credentials_fn=_NO_CREDENTIALS,
    )
    assert len(df) == 1


def test_partially_cached_range_only_fetches_missing_days(tmp_path):
    # Warm the cache for Friday only.
    fri_bars = [_bar("2026-01-02T14:30:00Z", 1, 1, 1, 1, 100)]
    fetch_intraday_bars(
        "QQQ", "2026-01-02", "2026-01-02",
        cache_dir=str(tmp_path), session=_FakeSession([_page(fri_bars)]),
        credentials_fn=_REAL_CREDENTIALS,
    )

    # Now request Friday through Monday (skips the weekend) -- Friday is
    # cached, Monday is not, so the fetch must only need Monday's data.
    mon_bars = [_bar("2026-01-05T14:30:00Z", 2, 2, 2, 2, 200)]
    session = _FakeSession([_page(mon_bars)])
    df = fetch_intraday_bars(
        "QQQ", "2026-01-02", "2026-01-05",
        cache_dir=str(tmp_path), session=session, credentials_fn=_REAL_CREDENTIALS,
    )
    assert len(df) == 2
    assert len(session.calls) == 1
    # The fetch window must start at the missing day (Monday), not
    # re-request the already-cached Friday.
    assert session.calls[0]["params"]["start"].startswith("2026-01-05")


# ---------------------------------------------------------------------------
# Weekend handling
# ---------------------------------------------------------------------------

def test_weekend_days_are_not_treated_as_gaps(tmp_path):
    # Friday 2026-01-02 through Monday 2026-01-05 -- only 2 business days.
    bars = [
        _bar("2026-01-02T14:30:00Z", 1, 1, 1, 1, 100),
        _bar("2026-01-05T14:30:00Z", 2, 2, 2, 2, 200),
    ]
    session = _FakeSession([_page(bars)])
    df = fetch_intraday_bars(
        "QQQ", "2026-01-02", "2026-01-05",
        cache_dir=str(tmp_path), session=session, credentials_fn=_REAL_CREDENTIALS,
    )
    assert len(df) == 2
    trading_dates = sorted({ts.date().isoformat() for ts in df.index})
    assert trading_dates == ["2026-01-02", "2026-01-05"]


def test_real_nyse_holiday_inside_range_does_not_raise_gap_error(tmp_path):
    # Guards against treating a real NYSE holiday as a data gap:
    # pd.bdate_range only excludes Saturday/Sunday, so a gap check built on
    # it would raise a spurious IntradayDataGapError for a request spanning
    # a real holiday, breaking the client on essentially any realistic
    # multi-month historical range. 2023-11-23 is Thanksgiving, a real NYSE
    # closure (pandas_market_calendars.get_calendar("NYSE").schedule(...)
    # omits 2023-11-23 and includes 2023-11-24, the half-day after).
    open_days = ["2023-11-20", "2023-11-21", "2023-11-22", "2023-11-24"]
    bars = [_bar(f"{day}T14:30:00Z", 1, 1, 1, 1, 100) for day in open_days]
    session = _FakeSession([_page(bars)])

    df = fetch_intraday_bars(
        "QQQ", "2023-11-20", "2023-11-24",
        cache_dir=str(tmp_path), session=session, credentials_fn=_REAL_CREDENTIALS,
    )

    trading_dates = sorted({ts.date().isoformat() for ts in df.index})
    assert trading_dates == open_days
    assert "2023-11-23" not in trading_dates


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

def test_pagination_follows_next_page_token_until_null(tmp_path):
    page1 = _page([_bar("2026-01-02T14:30:00Z", 1, 1, 1, 1, 100)], next_page_token="tok-1")
    page2 = _page([_bar("2026-01-02T14:31:00Z", 2, 2, 2, 2, 200)], next_page_token=None)
    session = _FakeSession([page1, page2])

    df = fetch_intraday_bars(
        "QQQ", "2026-01-02", "2026-01-02",
        cache_dir=str(tmp_path), session=session, credentials_fn=_REAL_CREDENTIALS,
    )

    assert len(df) == 2
    assert len(session.calls) == 2
    assert "page_token" not in session.calls[0]["params"]
    assert session.calls[1]["params"]["page_token"] == "tok-1"


def test_pagination_chain_that_never_terminates_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(intraday_data, "MAX_PAGES_PER_REQUEST", 3)

    class _InfiniteSession:
        def __init__(self):
            self.calls = 0

        def get(self, url, params=None, headers=None, timeout=None, allow_redirects=None):
            self.calls += 1
            return _FakeResponse(_page([], next_page_token=f"tok-{self.calls}"))

    session = _InfiniteSession()
    with pytest.raises(PaginationError):
        fetch_intraday_bars(
            "QQQ", "2026-01-02", "2026-01-02",
            cache_dir=str(tmp_path), session=session, credentials_fn=_REAL_CREDENTIALS,
        )
    assert session.calls <= 4  # bounded, never looped unboundedly


# ---------------------------------------------------------------------------
# Fail-closed on missing data
# ---------------------------------------------------------------------------

def test_empty_day_response_raises_gap_error_not_fabricated_data(tmp_path):
    session = _FakeSession([_page([])])
    with pytest.raises(IntradayDataGapError):
        fetch_intraday_bars(
            "QQQ", "2026-01-02", "2026-01-02",
            cache_dir=str(tmp_path), session=session, credentials_fn=_REAL_CREDENTIALS,
        )
    # And nothing must have been cached for the failed day, under either a
    # flat (symbol, day) path or the real (symbol, adjustment, day) path.
    assert not os.path.exists(os.path.join(str(tmp_path), "QQQ", "2026-01-02.csv"))
    assert not os.path.exists(os.path.join(str(tmp_path), "QQQ", "split", "2026-01-02.csv"))


def test_null_bars_field_treated_as_empty_not_malformed(tmp_path):
    # Observed against the real Alpaca API: a date range
    # before the vendor's own coverage returns HTTP 200 with
    # {"bars": null, ...} -- not an empty list, a real null. This must be
    # treated the same as zero bars (a gap), not raise a generic malformed-
    # response error.
    session = _FakeSession([_page(None)])
    with pytest.raises(IntradayDataGapError):
        fetch_intraday_bars(
            "QQQ", "2026-01-02", "2026-01-02",
            cache_dir=str(tmp_path), session=session, credentials_fn=_REAL_CREDENTIALS,
        )


def test_bars_entirely_outside_rth_also_raise_gap_error(tmp_path):
    # A response with real bars, but none inside the RTH window, is still a
    # gap for our purposes -- we need usable RTH data, not just any bars.
    bars = [_bar("2026-01-02T09:00:00Z", 1, 1, 1, 1, 100)]  # 04:00 ET, pre-market
    session = _FakeSession([_page(bars)])
    with pytest.raises(IntradayDataGapError):
        fetch_intraday_bars(
            "QQQ", "2026-01-02", "2026-01-02",
            cache_dir=str(tmp_path), session=session, credentials_fn=_REAL_CREDENTIALS,
        )


# ---------------------------------------------------------------------------
# Credential injection
# ---------------------------------------------------------------------------

def test_missing_credentials_fail_before_any_http_call(tmp_path):
    with pytest.raises(CredentialError):
        fetch_intraday_bars(
            "QQQ", "2026-01-02", "2026-01-02",
            cache_dir=str(tmp_path), session=_NoCallSession(), credentials_fn=_NO_CREDENTIALS,
        )


def test_credentials_sent_as_alpaca_headers(tmp_path):
    bars = [_bar("2026-01-02T14:30:00Z", 1, 1, 1, 1, 100)]
    session = _FakeSession([_page(bars)])
    fetch_intraday_bars(
        "QQQ", "2026-01-02", "2026-01-02",
        cache_dir=str(tmp_path), session=session,
        credentials_fn=lambda: ("KEYID123", "SECRET456"),
    )
    assert session.calls[0]["headers"]["APCA-API-KEY-ID"] == "KEYID123"
    assert session.calls[0]["headers"]["APCA-API-SECRET-KEY"] == "SECRET456"


# ---------------------------------------------------------------------------
# Retry / backoff on 429 and 5xx
# ---------------------------------------------------------------------------

def test_retries_on_429_then_succeeds(tmp_path):
    bars = [_bar("2026-01-02T14:30:00Z", 1, 1, 1, 1, 100)]
    session = _FakeSession([
        _FakeResponse({}, status_code=429),
        _FakeResponse(_page(bars)),
    ])
    sleeps = []
    df = fetch_intraday_bars(
        "QQQ", "2026-01-02", "2026-01-02",
        cache_dir=str(tmp_path), session=session, credentials_fn=_REAL_CREDENTIALS,
        sleep_fn=sleeps.append,
    )
    assert len(df) == 1
    assert len(session.calls) == 2
    assert len(sleeps) == 1


def test_retry_budget_exhausted_raises(tmp_path):
    session = _FakeSession([_FakeResponse({}, status_code=503) for _ in range(10)])
    with pytest.raises(AlpacaRequestError):
        fetch_intraday_bars(
            "QQQ", "2026-01-02", "2026-01-02",
            cache_dir=str(tmp_path), session=session, credentials_fn=_REAL_CREDENTIALS,
            sleep_fn=lambda _seconds: None,
        )


def test_terminal_4xx_status_does_not_retry(tmp_path):
    session = _FakeSession([_FakeResponse({}, status_code=403)])
    with pytest.raises(AlpacaRequestError):
        fetch_intraday_bars(
            "QQQ", "2026-01-02", "2026-01-02",
            cache_dir=str(tmp_path), session=session, credentials_fn=_REAL_CREDENTIALS,
        )
    assert len(session.calls) == 1


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------

def test_request_params_match_alpaca_bars_contract(tmp_path):
    bars = [_bar("2026-01-02T14:30:00Z", 1, 1, 1, 1, 100)]
    session = _FakeSession([_page(bars)])
    fetch_intraday_bars(
        "QQQ", "2026-01-02", "2026-01-02",
        cache_dir=str(tmp_path), session=session, credentials_fn=_REAL_CREDENTIALS,
    )
    params = session.calls[0]["params"]
    assert params["timeframe"] == "1Min"
    assert params["feed"] == "sip"
    assert params["sort"] == "asc"
    assert params["limit"] == 10_000
    assert "adjustment" in params
    assert session.calls[0]["url"] == "https://data.alpaca.markets/v2/stocks/QQQ/bars"


def test_adjustment_defaults_to_split_and_is_overridable(tmp_path):
    # Observed on vendor data: a real NVDA 10-for-1 split shows a ~10x
    # discontinuity under
    # adjustment="raw" but not under "split" -- "split" is the correct
    # default for a client that will hold positions across corporate
    # actions. This locks that default in as a regression check.
    bars = [_bar("2026-01-02T14:30:00Z", 1, 1, 1, 1, 100)]

    default_session = _FakeSession([_page(bars)])
    fetch_intraday_bars(
        "QQQ", "2026-01-02", "2026-01-02",
        cache_dir=str(tmp_path / "default"), session=default_session,
        credentials_fn=_REAL_CREDENTIALS,
    )
    assert default_session.calls[0]["params"]["adjustment"] == "split"

    override_session = _FakeSession([_page(bars)])
    fetch_intraday_bars(
        "QQQ", "2026-01-02", "2026-01-02",
        cache_dir=str(tmp_path / "override"), session=override_session,
        credentials_fn=_REAL_CREDENTIALS, adjustment="raw",
    )
    assert override_session.calls[0]["params"]["adjustment"] == "raw"


# ---------------------------------------------------------------------------
# Cache keying and cache-path safety
# ---------------------------------------------------------------------------

def test_different_adjustment_values_do_not_share_a_cache_entry(tmp_path):
    # Cache files are keyed by (symbol, adjustment, day), not only by
    # (symbol, day). A day cached under adjustment="split" must NOT be
    # silently served back when the SAME symbol/day is later requested
    # with adjustment="raw" -- those are genuinely different bar values
    # across any corporate action in that window, and a shared cache_dir
    # (the realistic scenario -- e.g. DEFAULT_CACHE_DIR) must not collide.
    split_bars = [_bar("2026-01-02T14:30:00Z", 100, 100, 100, 100, 100)]
    raw_bars = [_bar("2026-01-02T14:30:00Z", 1000, 1000, 1000, 1000, 100)]
    shared_cache_dir = str(tmp_path)  # deliberately the SAME cache_dir both times

    split_session = _FakeSession([_page(split_bars)])
    split_df = fetch_intraday_bars(
        "QQQ", "2026-01-02", "2026-01-02",
        cache_dir=shared_cache_dir, session=split_session,
        credentials_fn=_REAL_CREDENTIALS, adjustment="split",
    )
    assert len(split_session.calls) == 1
    assert split_df["Close"].iloc[0] == 100

    raw_session = _FakeSession([_page(raw_bars)])
    raw_df = fetch_intraday_bars(
        "QQQ", "2026-01-02", "2026-01-02",
        cache_dir=shared_cache_dir, session=raw_session,
        credentials_fn=_REAL_CREDENTIALS, adjustment="raw",
    )
    # The second call, with a DIFFERENT adjustment over the SAME cache_dir,
    # must actually hit the network -- not silently return the "split"
    # cache entry -- and must return data consistent with adjustment="raw".
    assert len(raw_session.calls) == 1
    assert raw_df["Close"].iloc[0] == 1000

    # And the original "split" entry must still be intact/unclobbered.
    replay_session = _NoCallSession()
    replay_df = fetch_intraday_bars(
        "QQQ", "2026-01-02", "2026-01-02",
        cache_dir=shared_cache_dir, session=replay_session,
        credentials_fn=_REAL_CREDENTIALS, adjustment="split",
    )
    assert replay_df["Close"].iloc[0] == 100


def test_invalid_adjustment_value_is_rejected(tmp_path):
    # adjustment flows into the cache PATH (not just request params),
    # so it needs the same enum discipline as symbol -- Alpaca's real
    # documented values are raw/split/dividend/all.
    with pytest.raises(ValueError):
        fetch_intraday_bars(
            "QQQ", "2026-01-02", "2026-01-02",
            cache_dir=str(tmp_path), session=_NoCallSession(),
            credentials_fn=_REAL_CREDENTIALS, adjustment="../../evil",
        )
    assert list(pathlib.Path(str(tmp_path)).iterdir()) == []


# ---------------------------------------------------------------------------

def test_symbol_with_path_traversal_segments_raises_valueerror(tmp_path):
    # The symbol is interpolated into a filesystem path, so a
    # non-emptiness check is not enough: a symbol containing
    # path-traversal segments must be rejected before any path is built or
    # any file is written, not merely fail later for an unrelated reason.
    outside_marker = pathlib.Path(str(tmp_path)).parent.parent / "evil"
    with pytest.raises(ValueError):
        fetch_intraday_bars(
            "../../evil", "2026-01-02", "2026-01-02",
            cache_dir=str(tmp_path), session=_NoCallSession(), credentials_fn=_REAL_CREDENTIALS,
        )
    assert not outside_marker.exists()
    # cache_dir itself must remain completely untouched -- validation must
    # happen before any os.makedirs/os.path.join-based write.
    assert list(pathlib.Path(str(tmp_path)).iterdir()) == []


def test_symbol_with_invalid_characters_raises_valueerror(tmp_path):
    with pytest.raises(ValueError):
        fetch_intraday_bars(
            "QQQ; rm -rf /", "2026-01-02", "2026-01-02",
            cache_dir=str(tmp_path), session=_NoCallSession(), credentials_fn=_REAL_CREDENTIALS,
        )


def test_symbol_equal_to_double_dot_is_rejected_despite_matching_character_class(tmp_path):
    # ".." is built entirely from otherwise-allowed characters (each "."
    # is a legitimate share-class separator, e.g. "BRK.B") -- a naive
    # character-class allowlist alone would NOT catch this classic
    # traversal token on its own. Confirm it is explicitly rejected too.
    with pytest.raises(ValueError):
        fetch_intraday_bars(
            "..", "2026-01-02", "2026-01-02",
            cache_dir=str(tmp_path), session=_NoCallSession(), credentials_fn=_REAL_CREDENTIALS,
        )


def test_valid_symbol_with_share_class_dot_is_accepted(tmp_path):
    bars = [_bar("2026-01-02T14:30:00Z", 1, 1, 1, 1, 100)]
    session = _FakeSession([_page(bars)])
    df = fetch_intraday_bars(
        "BRK.B", "2026-01-02", "2026-01-02",
        cache_dir=str(tmp_path), session=session, credentials_fn=_REAL_CREDENTIALS,
    )
    assert len(df) == 1


# ---------------------------------------------------------------------------
# Network exceptions, atomic cache writes and redirects
# ---------------------------------------------------------------------------

def test_network_exception_during_request_retries_then_succeeds(tmp_path):
    # session.get() can raise a real requests exception directly
    # (a transient connection blip), not just return a 429/5xx response.
    # That must retry through the same path, not crash the caller with a
    # raw `requests` exception.
    bars = [_bar("2026-01-02T14:30:00Z", 1, 1, 1, 1, 100)]

    class _FlakyNetworkSession:
        def __init__(self):
            self.calls = 0

        def get(self, url, params=None, headers=None, timeout=None, allow_redirects=None):
            self.calls += 1
            if self.calls == 1:
                raise requests.exceptions.ConnectionError("simulated transient network blip")
            return _FakeResponse(_page(bars))

    session = _FlakyNetworkSession()
    sleeps = []
    df = fetch_intraday_bars(
        "QQQ", "2026-01-02", "2026-01-02",
        cache_dir=str(tmp_path), session=session, credentials_fn=_REAL_CREDENTIALS,
        sleep_fn=sleeps.append,
    )
    assert len(df) == 1
    assert session.calls == 2
    assert len(sleeps) == 1


def test_network_exception_retry_budget_exhausted_raises_alpaca_error(tmp_path):
    class _AlwaysFailsSession:
        def __init__(self):
            self.calls = 0

        def get(self, url, params=None, headers=None, timeout=None, allow_redirects=None):
            self.calls += 1
            raise requests.exceptions.Timeout("simulated persistent timeout")

    session = _AlwaysFailsSession()
    with pytest.raises(AlpacaRequestError):
        fetch_intraday_bars(
            "QQQ", "2026-01-02", "2026-01-02",
            cache_dir=str(tmp_path), session=session, credentials_fn=_REAL_CREDENTIALS,
            sleep_fn=lambda _seconds: None,
        )
    # attempts 0..MAX_RETRIES inclusive = MAX_RETRIES + 1 real calls, then
    # the budget is exhausted -- never an unbounded retry loop.
    assert session.calls == intraday_data.MAX_RETRIES + 1


# ---------------------------------------------------------------------------

def test_interrupted_cache_write_leaves_no_partial_file_and_prior_entries_intact(tmp_path, monkeypatch):
    # _write_cache must not write directly to the final path: a process
    # killed mid-write (or a race) could then leave a partial CSV that a
    # later call would treat as a valid cache hit ("file exists" == "trust it").
    original_to_csv = pd.DataFrame.to_csv
    state = {"calls": 0}

    def _flaky_to_csv(self, path_or_buf=None, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            return original_to_csv(self, path_or_buf, **kwargs)
        # Simulate a process killed mid-write: some bytes land on disk,
        # then the write is interrupted before it can complete.
        with open(path_or_buf, "w", encoding="utf-8") as handle:
            handle.write("timestamp,Open,High,Low\n2026-01-05 09:30:00-05:00,1,1")
        raise OSError("simulated crash mid-write")

    monkeypatch.setattr(pd.DataFrame, "to_csv", _flaky_to_csv)

    bars = [
        _bar("2026-01-02T14:30:00Z", 1, 1, 1, 1, 100),
        _bar("2026-01-05T14:30:00Z", 2, 2, 2, 2, 200),
    ]
    session = _FakeSession([_page(bars)])
    with pytest.raises(OSError):
        fetch_intraday_bars(
            "QQQ", "2026-01-02", "2026-01-05",
            cache_dir=str(tmp_path), session=session, credentials_fn=_REAL_CREDENTIALS,
        )

    day1_path = os.path.join(str(tmp_path), "QQQ", "split", "2026-01-02.csv")
    day2_path = os.path.join(str(tmp_path), "QQQ", "split", "2026-01-05.csv")

    # Day 1's write completed normally (real to_csv + atomic replace)
    # before the simulated crash on day 2 -- it must remain a complete,
    # readable file, untouched by the later failure.
    assert os.path.exists(day1_path)
    reread = pd.read_csv(day1_path)
    assert len(reread) == 1

    # Day 2's write crashed mid-write. Atomic replace means the FINAL path
    # is either absent entirely or fully valid -- never partial. Here it
    # must be absent, since the crash happened before os.replace() ran.
    assert not os.path.exists(day2_path)

    # And no stray temp file left lying around either -- _write_cache's
    # own cleanup on failure must remove it.
    symbol_adj_dir = os.path.dirname(day2_path)
    leftovers = [f for f in os.listdir(symbol_adj_dir) if f != "2026-01-02.csv"]
    assert leftovers == []


# ---------------------------------------------------------------------------

def test_requests_are_made_with_allow_redirects_false(tmp_path):
    bars = [_bar("2026-01-02T14:30:00Z", 1, 1, 1, 1, 100)]
    session = _FakeSession([_page(bars)])
    fetch_intraday_bars(
        "QQQ", "2026-01-02", "2026-01-02",
        cache_dir=str(tmp_path), session=session, credentials_fn=_REAL_CREDENTIALS,
    )
    assert session.calls[0]["allow_redirects"] is False


def test_redirect_response_is_hard_failure_not_followed(tmp_path):
    # Custom headers like APCA-API-KEY-ID/APCA-API-SECRET-KEY are not
    # stripped by `requests` on a cross-origin redirect the way the
    # standard Authorization header is -- never auto-follow, and never
    # manually follow either. A 3xx is a hard failure.
    session = _FakeSession([_FakeResponse({}, status_code=302)])
    with pytest.raises(AlpacaRequestError):
        fetch_intraday_bars(
            "QQQ", "2026-01-02", "2026-01-02",
            cache_dir=str(tmp_path), session=session, credentials_fn=_REAL_CREDENTIALS,
        )
    # Exactly one call was made -- no second request attempting to
    # "follow" the redirect location.
    assert len(session.calls) == 1
