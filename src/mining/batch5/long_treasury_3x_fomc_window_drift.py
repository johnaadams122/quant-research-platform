"""Batch5 candidate: long_treasury_3x_fomc_window_drift (single_long).

3x long 20+ year Treasury (TMF) held ONLY in the three-session window
around each SCHEDULED FOMC statement: the session before the statement,
the statement session, and the session after (Hillenbrand, RFS 2025;
Savor & Wilson, JFQA 2013). Flat otherwise. Rule, verbatim from the
curated candidate's rule_pseudocode:

    E = sorted list of SCHEDULED FOMC statement sessions (two-day meeting
        -> second day; one-day meeting -> that day), from federalreserve.gov
        calendars; unscheduled/emergency meetings excluded.
    For each NYSE session t in TMF's history (index s(t)):
        signal(t) = 1 if there exists e in E with s(t) in
                    {s(e)-2, s(e)-1, s(e)}  else 0
    signal(t)=1 at the close of t -> long TMF from the open of t+1; held
    while signal stays 1; exit at the close of the first session whose
    signal is 0 -> held from the open of e-1 through the close of e+1
    (exactly 3 sessions), one trade per scheduled meeting.

With the candidate's parameters window_sessions_before=1 and
window_sessions_after=1 the signal-1 positions are s(e)-before-1 .. s(e)+
after-1 = {s(e)-2, s(e)-1, s(e)}; hold_sessions=3 is checked at import to
equal before + 1 + after. No price-based input of any kind.

Event data: data/mining/batch5_events/long_treasury_3x_fomc_window_drift/
events.json ({"TMF": [ISO dates]}) with provenance.json alongside -- fetched
ONCE from the Federal Reserve's own pages (fomchistorical<year>.htm for
2009-2020, fomccalendars.htm for 2021 onward), headings for conference
calls, "(unscheduled)", "(cancelled)" and "(notation vote)" items excluded.
Cross-checked against the repo's data/mining/fomc_statement_dates.json: the
only differences are that file's two 2020 emergency statements
(2020-03-03, 2020-03-15), which this rule excludes by parameter.

Design (the four non-negotiable properties):

* NO LOOK-AHEAD -- the signal is a pure function of the session CALENDAR and
  a calendar of statement dates every one of which the Fed published months
  in advance; no price at any date enters. The whole boolean series is built
  ONCE in the factory over the traded calendar truncated to the bucket's last
  bar, reindexed to ``index``, and looked up by position.
* PERFORMANCE -- the returned signal_fn does one array lookup plus one scalar
  date check per call; nothing is recomputed over the growing slice.
* HOLDOUT-BLIND -- the traded calendar is truncated to <= the last bar of
  ``index`` before positions are resolved; bars BEFORE the bucket are kept so
  a statement whose pre-window sessions precede the bucket still resolves.
* DIRECTION -- always long TMF; there is no short side to this rule.

Interpretation choices (the pseudocode leaves these to the implementer):

1. s(e) is resolved on TMF's OWN bar calendar (the frame's dates), which is
   the pseudocode's "NYSE session t in TMF's history". A statement date that
   is not a bar in that calendar (never the case on real data -- the FOMC
   does not release statements on market holidays) anchors to the first bar
   on or after it, the reading under which a statement released on a closed
   day is traded on the next session. A statement dated BEFORE the first bar
   of TMF's history (all pre-inception meetings) is not anchored at all --
   it belongs to no session of TMF's history and is simply skipped.
2. A statement whose anchored session lies beyond the bucket's last bar
   produces no signal inside the bucket: the truncated calendar cannot place
   it, and the only bars it could mark are the bucket's final one or two,
   where the engine would open a position it force-closes at end_of_data
   after zero or one session -- not the rule's 3-session hold. A statement
   ON the bucket's last bar does mark its window (positions exist).
3. A statement in the first two bars of TMF's history marks only the
   positions that exist (clipped at 0); the engine then holds a shorter
   first trade, exactly what the calendar admits.
4. A bucket (holdout) starting inside a window sees the window's remaining
   signal-1 bars and the engine enters at the next open -- the ordinary
   bucket-boundary effect every calendar rule in this repo has; nothing is
   special-cased.
5. Scheduled meetings are >= 5 weeks apart, so windows never overlap and
   no merge logic is needed.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "long_treasury_3x_fomc_window_drift"
SHAPE = "single_long"
TRADED = ["TMF"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "TMF"
# Copied byte-for-byte from the candidate JSON "parameters" (driver drift check).
PARAMETERS = {
    "event": "scheduled_fomc_statement_session",
    "statement_session_rule": "second day of a two-day meeting; the meeting day for a one-day meeting (statement-day convention since 1994 per Hillenbrand)",
    "include_unscheduled_meetings": False,
    "window_sessions_before": 1,
    "window_sessions_after": 1,
    "hold_sessions": 3,
    "event_source": (
        "federalreserve.gov FOMC calendars: fomc_historical_year.htm (by "
        "year, through 2020) and fomccalendars.htm (2021-2027); the "
        "statement-date list must cover 2009-04 (TMF inception) onward and "
        "be filtered to scheduled meetings -- a free, public, point-in-time "
        "source"
    ),
}
EXPECTED_MIN_TRADES = 100

_BEFORE = int(PARAMETERS["window_sessions_before"])
_AFTER = int(PARAMETERS["window_sessions_after"])
if _BEFORE + 1 + _AFTER != int(PARAMETERS["hold_sessions"]):
    raise ValueError(f"{FAMILY_ID}: hold_sessions must equal "
                     "window_sessions_before + 1 + window_sessions_after")
if PARAMETERS["include_unscheduled_meetings"] is not False:
    raise ValueError(f"{FAMILY_ID}: only the scheduled-meetings calendar is built")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EVENTS_PATH = os.path.join(_REPO_ROOT, "data", "mining", "batch5_events",
                           FAMILY_ID, "events.json")


def _naive_index(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_localize(None)
    return out.normalize()


def load_statement_dates(path: str | None = None) -> pd.DatetimeIndex:
    """Scheduled FOMC statement sessions from the events file (sorted,
    unique, tz-naive). Raises on a malformed file rather than guessing."""
    path = EVENTS_PATH if path is None else path
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    dates = payload.get(TRADED[0]) if isinstance(payload, dict) else None
    if not isinstance(dates, list) or not dates:
        raise ValueError(f"{FAMILY_ID}: {path} must map {TRADED[0]!r} to a "
                         "non-empty list of ISO dates")
    parsed = pd.DatetimeIndex(pd.to_datetime(dates, format="%Y-%m-%d"))
    if parsed.has_duplicates or not parsed.is_monotonic_increasing:
        raise ValueError(f"{FAMILY_ID}: statement dates must be sorted and unique")
    return parsed


def fomc_window_signal(sessions: pd.DatetimeIndex,
                       statement_dates: pd.DatetimeIndex) -> np.ndarray:
    """Boolean array over ``sessions`` (tz-naive, normalized, sorted,
    unique): True at positions s(e)-_BEFORE-1 .. s(e)+_AFTER-1 for every
    statement e that anchors to a session in ``sessions`` (docstring choices
    1-3). Pure calendar arithmetic; no price is read."""
    n = len(sessions)
    signal = np.zeros(n, dtype=bool)
    if n == 0 or len(statement_dates) == 0:
        return signal
    first_session = sessions[0]
    anchors = sessions.searchsorted(statement_dates, side="left")
    for e, pos in zip(statement_dates, anchors):
        if pos >= n:
            continue  # statement beyond the (truncated) calendar: choice 2
        if e < first_session:
            continue  # statement before the calendar begins: choice 1
        start = max(0, int(pos) - _BEFORE - 1)
        stop = min(n - 1, int(pos) + _AFTER - 1)
        signal[start:stop + 1] = True
    return signal


def make_signal_fn(ticker: str, index, frames, *, statement_dates=None):
    """Factory per the batch5 contract: ``index`` is the bucket the engine
    will run over; ``frames`` holds full-history OHLC for TRADED (+ SIGNAL,
    empty here). ``statement_dates`` is a test seam only -- the driver never
    passes it and the real events file is used."""
    if ticker != TRADED[0]:
        raise ValueError(f"{FAMILY_ID}: make_signal_fn expects ticker "
                         f"{TRADED[0]!r}, got {ticker!r}")
    frame = frames.get(ticker) if isinstance(frames, dict) else None
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{FAMILY_ID}: frames must contain a non-empty OHLC "
                         f"DataFrame for {ticker}")

    bucket = _naive_index(index)
    if len(bucket) == 0:
        raise ValueError(f"{FAMILY_ID}: index must not be empty")
    if not bucket.is_monotonic_increasing or bucket.has_duplicates:
        raise ValueError(f"{FAMILY_ID}: index must be sorted and unique")
    last_bar = bucket[-1]

    sessions = _naive_index(frame.index).unique().sort_values()
    sessions = sessions[sessions <= last_bar]  # holdout-blind truncation

    if statement_dates is None:
        statements = load_statement_dates()
    else:
        statements = _naive_index(statement_dates).unique().sort_values()

    signal_on_sessions = pd.Series(fomc_window_signal(sessions, statements),
                                   index=sessions)
    signal_arr = signal_on_sessions.reindex(
        bucket, fill_value=False).to_numpy(dtype=bool)
    bucket_values = bucket.asi8  # int64 ns for an O(1) scalar date check
    n_bucket = len(bucket)

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        if i < 0 or i >= n_bucket:
            raise IndexError(f"{FAMILY_ID}: bar position {i} outside the "
                             f"{n_bucket}-bar bucket this signal was built for")
        today = frame.index[-1]
        if pd.Timestamp(today).value != bucket_values[i]:
            raise ValueError(f"{FAMILY_ID}: bar {i} is {today}, but the bucket "
                             f"index has {bucket[i]} there -- the engine must "
                             "run over the same index handed to the factory")
        on = bool(signal_arr[i])
        return SignalResult(entry=on, exit=not on)

    return signal_fn
