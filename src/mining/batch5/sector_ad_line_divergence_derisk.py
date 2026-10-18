"""Batch5 candidate: sector_ad_line_divergence_derisk (shape panel_breadth).

Sector A/D-Line Bearish Divergence De-risk. Default state is long SPY. On any
session where SPY closes at or above its trailing 126-session closing high
while the cumulative sector advance/decline line is BELOW its own trailing
126-session high (an index high that breadth fails to confirm -- the
StockCharts ChartSchool definition), a divergence is flagged and the strategy
is flat for 21 sessions including the flag session; every new flag restarts
the clock. Then it re-enters. No stops, no other exits, cash earns nothing.

Breadth source: the pipeline's pinned ``net_advance_ratio`` breadth snapshot on
the point-in-time ``sector_spdr_11`` basis (advancers minus decliners over the
resolved seasoned members, >= 9 resolved members else the session is unknown),
built ONLY via ``scripts/build_sector_panel.py snapshot`` and loaded through
``panel.load_breadth_snapshot`` against the sha pinned in ``SNAPSHOTS``. The
A/D line is the running sum of that ratio over the snapshot's contiguous
defined span, starting at its first defined session (the snapshot's own
provenance records zero broken and zero omitted sessions, and the sibling
path-dependent snapshots built on the same panel would have been refused
otherwise).

Rule (rule_pseudocode, verbatim in spirit), evaluated at the close of bar t:
    nar_t    = net_advance_ratio as of t
    adl_t    = cumsum(nar) over the defined span from the first defined session
    spy_hi   = close_SPY(t) >= max(close_SPY(t-125..t))    # 126 incl. today
    adl_hi   = adl_t >= max(adl(t-125..t))                 # 126 incl. today
    flag     = spy_hi AND NOT adl_hi
    off      = any flag in t-20..t                          # 21 incl. today
    long SPY when off is False; flat when off is True.
Engine mapping: entry = long-condition holds (fills at the NEXT open), exit =
off (fills at THIS close), which is exactly "flat->long buys at open(t+1);
long->flat exits at the close of the session on which off first becomes True".

Interpretation choices (the pseudocode leaves these open; each picks the
reading closest to the cited ChartSchool definition and invents no parameter):
  1. ``adl_hi`` before 126 DEFINED A/D sessions exist. The pseudocode says only
     that ``spy_hi`` is False before 126 sessions exist. A divergence is a
     disagreement between two highs, so while the A/D line's own 126-session
     high is undefined no divergence can be established: ``flag`` is False
     there rather than True-by-default (the alternative would flag every
     early SPY high as "unconfirmed" purely because the A/D history is short).
  2. Bars before the breadth series is defined at all (SPY trades from 1993,
     the panel from 1998-12-23). The rule is undefined there, so the module
     takes NO position (entry False, exit False), matching
     ``signals.make_external_regime_signal_fn``'s "a bar before the signal's
     first valid observation is a no-op". The default long state applies only
     where the breadth that could switch it off actually exists.
  3. The A/D line is accumulated on the traded frame's calendar via one
     ``as_of`` lookup per SPY bar. ``AsOfSeries`` exposes no dates, so the
     factory proves the two calendars are one-to-one over the snapshot's span
     (the frame carries a bar ON the first defined session, and the count of
     frame bars in [first_date, last_date] equals ``len(series)``) and refuses
     otherwise -- a forward-filled duplicate or a skipped session would
     otherwise silently distort the line.
  4. ``SIGNAL`` is empty: the eleven sector SPDRs enter only through the
     sha-pinned snapshot (the point-in-time panel), never as raw OHLC frames.

Performance: every series is precomputed once in the factory (vectorised
rolling/cumsum on the traded frame truncated to the bucket's last bar), then
reindexed onto the bucket; ``signal_fn`` is two array reads plus one O(1)
alignment check per bar.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from mining import panel
from mining.backtest_engine import SignalResult

FAMILY_ID = "sector_ad_line_divergence_derisk"
SHAPE = "panel_breadth"
TRADED = ["SPY"]
SIGNAL = []
SNAPSHOTS = {
    "net_advance_ratio": {
        "path": ("data/mining/panel/snapshots/"
                 "sector_spdr_11_net_advance_ratio_m9_2026-09-04.json"),
        "sha256": ("02be5e15a821c6a727cee6b84f35161380ba504b1a9a805caf06a95e"
                   "195bdb52"),
        "indicator": "net_advance_ratio",
        "params": {"min_members": 9},
    },
}
BENCHMARK = "SPY"
# Copied byte-for-byte from the curated candidate record ("parameters"); the driver
# rejects any drift under json.dumps(sort_keys=True).
PARAMETERS = {
    "indicator": "net_advance_ratio",
    "min_members": 9,
    "ad_line": ("cumulative sum of net_advance_ratio over the contiguous "
                "defined span from the snapshot's first defined session "
                "(interior gap refused)"),
    "high_lookback_sessions": 126,
    "spy_high_test": "close >= trailing 126-session max including today",
    "adl_confirmation_test": ("ad_line >= its trailing 126-session max "
                              "including today"),
    "flat_sessions_after_flag": 21,
    "default_state": "long_SPY",
    "flat_asset": "cash",
    "execution": "next_open",
}
EXPECTED_MIN_TRADES = 60

_HIGH_LOOKBACK = int(PARAMETERS["high_lookback_sessions"])
_FLAT_SESSIONS = int(PARAMETERS["flat_sessions_after_flag"])

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def load_breadth() -> panel.AsOfSeries:
    """Load the pinned net_advance_ratio snapshot (sha-verified) as an
    as-of-only handle. Raises on any drift from the pinned bytes."""
    spec = SNAPSHOTS["net_advance_ratio"]
    path = spec["path"]
    if not os.path.isabs(path):
        path = os.path.join(_REPO_ROOT, path)
    return panel.load_breadth_snapshot(path, expect_sha256=spec["sha256"])


def _check_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{FAMILY_ID}: traded frame must be a non-empty DataFrame")
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:
        raise ValueError(f"{FAMILY_ID}: traded frame index must be tz-naive")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError(f"{FAMILY_ID}: traded frame index must be sorted unique")
    if "Close" not in frame.columns:
        raise ValueError(f"{FAMILY_ID}: traded frame needs a Close column")
    return frame


def _check_calendar_one_to_one(frame_index: pd.DatetimeIndex,
                               breadth: panel.AsOfSeries) -> None:
    """The A/D line is a cumsum, so each snapshot session must map to exactly
    one traded bar. AsOfSeries exposes no dates, only ``describe()`` and
    ``as_of``; this is the exact check those two surfaces allow."""
    described = breadth.describe()
    first = pd.Timestamp(described["first_date"])
    last = pd.Timestamp(described["last_date"])
    if frame_index[0] > first or first not in frame_index:
        raise ValueError(
            f"{FAMILY_ID}: the traded frame has no bar on the snapshot's first "
            f"defined session {first.date()} (frame starts "
            f"{frame_index[0].date()}); the A/D line must accumulate from the "
            "snapshot's first defined session, so a later start would be a "
            "different line")
    in_span = int(((frame_index >= first) & (frame_index <= last)).sum())
    if in_span != len(breadth):
        raise ValueError(
            f"{FAMILY_ID}: traded bars and snapshot sessions are not one-to-one "
            f"over {first.date()}..{last.date()}: {in_span} bars vs "
            f"{len(breadth)} observations. A forward-filled duplicate or a "
            "skipped session would silently distort the cumulative A/D line.")


def compute_state(frame: pd.DataFrame, breadth: panel.AsOfSeries, *,
                  check_calendar: bool = True) -> pd.DataFrame:
    """Every rule series on ``frame``'s own calendar, one row per bar.

    Columns: nar, adl, spy_hi, adl_hi, flag, off, entry, exit. Each value at
    date t depends only on data dated <= t (rolling/cumsum, never centred).
    ``check_calendar=False`` is for a caller that has ALREADY proven the
    full frame one-to-one and now passes a truncated prefix of it.
    """
    frame = _check_frame(frame)
    index = pd.DatetimeIndex(frame.index)
    if check_calendar:
        _check_calendar_one_to_one(index, breadth)

    nar = np.full(len(index), np.nan)
    for position, stamp in enumerate(index):
        value = breadth.as_of(stamp)
        if value is not None:
            nar[position] = value
    defined = ~np.isnan(nar)
    if not defined.any():
        raise ValueError(f"{FAMILY_ID}: breadth is undefined on every traded bar")
    first_defined = int(np.argmax(defined))
    if not defined[first_defined:].all():
        # unreachable with a sorted as_of series (forward-fill), kept fail-closed
        raise ValueError(f"{FAMILY_ID}: breadth has an interior undefined bar")

    adl = np.full(len(index), np.nan)
    adl[first_defined:] = np.cumsum(nar[first_defined:])
    adl_s = pd.Series(adl, index=index)
    close = frame["Close"].astype(float)

    spy_max = close.rolling(_HIGH_LOOKBACK, min_periods=_HIGH_LOOKBACK).max()
    spy_hi = (close >= spy_max) & spy_max.notna()
    adl_max = adl_s.rolling(_HIGH_LOOKBACK, min_periods=_HIGH_LOOKBACK).max()
    adl_hi = (adl_s >= adl_max) & adl_max.notna()
    flag = spy_hi & adl_max.notna() & ~adl_hi
    off = flag.astype(float).rolling(_FLAT_SESSIONS, min_periods=1).max() > 0.0
    defined_s = pd.Series(defined, index=index)
    entry = defined_s & ~off
    exit_ = defined_s & off
    return pd.DataFrame({
        "nar": nar, "adl": adl, "spy_hi": spy_hi.to_numpy(),
        "adl_hi": adl_hi.to_numpy(), "flag": flag.to_numpy(),
        "off": off.to_numpy(), "entry": entry.to_numpy(),
        "exit": exit_.to_numpy(),
    }, index=index)


def make_signal_fn(ticker: str, index, frames, *,
                   breadth: panel.AsOfSeries | None = None):
    """Driver entry point: ``make_signal_fn(ticker, index, frames)``.

    ``index`` is the bucket (in-sample or holdout) the engine will run over;
    ``frames[ticker]`` is the full-history OHLC frame. Precomputation uses the
    frame truncated to ``index[-1]`` (history before the bucket is warm-up;
    nothing after its last bar is read). ``breadth`` is injectable for
    synthetic tests; production loads the pinned snapshot.
    """
    if ticker != TRADED[0]:
        raise ValueError(f"{FAMILY_ID}: traded ticker must be {TRADED[0]!r}, "
                         f"got {ticker!r}")
    if ticker not in frames:
        raise ValueError(f"{FAMILY_ID}: frames has no {ticker!r} entry")
    bucket = pd.DatetimeIndex(index)
    if len(bucket) == 0:
        raise ValueError(f"{FAMILY_ID}: bucket index is empty")
    if bucket.tz is not None:
        raise ValueError(f"{FAMILY_ID}: bucket index must be tz-naive")
    if not bucket.is_monotonic_increasing or bucket.has_duplicates:
        raise ValueError(f"{FAMILY_ID}: bucket index must be sorted unique")
    frame = _check_frame(frames[ticker])
    series = breadth if breadth is not None else load_breadth()

    # The one-to-one calendar check reads dates only (never prices) over the
    # snapshot's whole span; the VALUES are computed on history through the
    # bucket's last bar only.
    _check_calendar_one_to_one(pd.DatetimeIndex(frame.index), series)
    history = frame.loc[frame.index <= bucket[-1]]
    state = compute_state(history, series, check_calendar=False)

    positions = state.index.get_indexer(bucket)
    if (positions < 0).any():
        missing = [str(stamp.date()) for stamp in bucket[positions < 0][:5]]
        raise ValueError(
            f"{FAMILY_ID}: bucket dates {missing} are not bars of the traded "
            "frame")
    entry_arr = state["entry"].to_numpy()[positions]
    exit_arr = state["exit"].to_numpy()[positions]
    n_bucket = len(bucket)

    def signal_fn(frame_slice: pd.DataFrame, i: int) -> SignalResult:
        if i < 0 or i >= n_bucket or frame_slice.index[-1] != bucket[i]:
            raise ValueError(
                f"{FAMILY_ID}: signal_fn slice does not end at bucket bar {i}")
        return SignalResult(entry=bool(entry_arr[i]), exit=bool(exit_arr[i]))

    return signal_fn
