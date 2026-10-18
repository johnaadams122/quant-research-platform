"""sector_breadth_thrust_zweig -- batch5 candidate (shape panel_breadth).

Zweig's breadth thrust (Zweig 1986; Hill 2019, StockCharts) computed on the
eleven-fund sector_spdr_11 panel: the 10-day exponential average of the share
of seasoned members that advanced on the day. A thrust fires when that
average moves from below 0.40 to above 0.615 within 10 sessions. On a thrust,
buy SPY at the next open and hold 63 sessions; no re-entry while a hold is
active. The ONLY signal input is the pipeline's sha-pinned net_advance_ratio
snapshot (data/mining/panel/snapshots/, see SNAPSHOTS); no sector fund's OHLC
is read here, so SIGNAL is empty and the eleven funds enter solely through
that snapshot.

Rule (verbatim from the curated candidate's rule_pseudocode)::

    nar_t = net_advance_ratio snapshot value = (advancers - decliners) /
            seasoned resolved members at t (>= 2 own bars; UNKNOWN if < 9
            resolved members).
    share_t = (1 + nar_t) / 2
    z_t = a*share_t + (1-a)*z_{t-1}, a = 2/11, seeded z = share at the first
          defined session; the EMA is computed over the contiguous defined
          span only (an interior unknown session is refused, as
          panel_signals does for path-dependent reducers).
    in_position = False
    for each session t with >= 11 prior defined sessions:
        if not in_position and z_t > 0.615 and min(z_{t-10} .. z_{t-1}) < 0.40:
            buy SPY at open(t+1); in_position = True
            exit_session = t+1 + 62          # 63 sessions in position
        elif in_position and t == exit_session:
            exit SPY at close(t); in_position = False
    no stops, no early exits, no re-entry during a hold; a new thrust during
    a hold is ignored.

Engine mapping (mining.backtest_engine.run_backtest semantics -- an entry
signalled on bar t fills at bar t+1's open, an exit signalled on bar t fills
at bar t's close):
    thrust bar t (flat)          -> entry=True  on index bar t
    bar t+63 of the bucket index -> exit=True   on that bar (63 sessions held,
                                    engine `held` = 62)
    every other bar              -> entry=False, exit=False
The whole entry/exit schedule is precomputed ONCE in the factory by replaying
the pseudocode's state machine over the bucket `index`, so the per-bar
signal_fn is two boolean array lookups plus one alignment guard.

Interpretation choices
----------------------
* "Session t" for the thrust test is a DEFINED session of the snapshot (the
  panel's own calendar); the 10 prior values z_{t-10}..z_{t-1} are the ten
  immediately preceding defined sessions. "At least 11 prior defined
  sessions" is applied literally: the earliest session that may fire is the
  twelfth defined session (0-based position 11), one later than the minimum
  the ten-value window itself would need.
* "open(t+1)" and the 63-session hold count on the TRADED bucket calendar
  (`index`, SPY's sessions). A thrust dated on a snapshot session that is
  not a bar of `index` is not acted on (never carried forward to a later
  bar); a thrust whose fill bar t+1 lies beyond `index` produces no entry
  (the engine could not fill it either); an exit beyond `index` is left to
  the engine's own end_of_data close. When the snapshot and SPY calendars
  coincide session-for-session, the first case never arises.
* The state machine starts FLAT at the first bar of each bucket, exactly as
  the engine does: a hold that began in the in-sample bucket does not
  suppress a thrust early in the holdout bucket, because the engine holds
  nothing there. (Per the pseudocode's `if`/`elif` order a thrust ON the
  exit session itself is ignored -- in_position is still True when the
  entry test runs -- and the schedule reproduces that.)
* EMA seed at the FIRST DEFINED SESSION of the snapshot (the candidate's own
  `ema_form`), not at the bucket's first bar. That needs the snapshot's
  values BEFORE the bucket for warm-up, so the factory looks up one value
  per defined snapshot session dated <= index[-1] (never later: the factory
  is holdout-blind). Values are read ONLY through AsOfSeries.as_of; the
  snapshot's calendar (its `dates` list) and its broken/omitted-session
  record are read from the same sha-verified file, after
  panel.load_breadth_snapshot has accepted it against the pinned sha.
* Refusal: any broken or omitted session recorded in the snapshot's
  provenance that lies inside the consulted span raises ValueError -- the
  path-dependent EMA may not bridge an unknown session (the same rule
  panel._accumulation_span applies). The real snapshot records none.
* Comparisons are strict as written: z_t > 0.615 and min(...) < 0.40.

Look-ahead: z_t uses snapshot values dated <= t (causal recursion), the
thrust test at t uses z_{t-10..t}, and the hold schedule at bar t depends
only on thrusts dated <= t. Nothing dated after index[-1] is consulted.

Performance: ~7,000 O(log n) as_of lookups plus one O(n) pass in the factory;
the per-bar signal_fn is O(1).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os

import numpy as np
import pandas as pd

from mining import panel
from mining.backtest_engine import SignalResult

FAMILY_ID = "sector_breadth_thrust_zweig"
SHAPE = "panel_breadth"
TRADED = ["SPY"]
SIGNAL = []  # the eleven sector funds enter only through the snapshot below
_SNAPSHOT_NAME = "sector_spdr_11_net_advance_ratio_m9_2026-09-04"
SNAPSHOTS = {
    _SNAPSHOT_NAME: {
        "path": f"data/mining/panel/snapshots/{_SNAPSHOT_NAME}.json",
        "sha256": "02be5e15a821c6a727cee6b84f35161380ba504b1a9a805caf06a95e195bdb52",
        "indicator": "net_advance_ratio",
        "params": {"min_members": 9},
    },
}
BENCHMARK = "SPY"
# Copied byte-for-byte from the curated candidate record
# (family sector_breadth_thrust_zweig, key "parameters").
PARAMETERS = {
    "indicator": "net_advance_ratio",
    "min_members": 9,
    "advancing_share": "(1 + net_advance_ratio) / 2",
    "ema_span": 10,
    "ema_form": "recursive, alpha = 2/(span+1), adjust=False, seeded at the first defined session",
    "setup_level": 0.4,
    "trigger_level": 0.615,
    "window_sessions": 10,
    "hold_sessions": 63,
    "reentry_during_hold": False,
    "execution": "next_open",
}
EXPECTED_MIN_TRADES = 50

# The pseudocode's literal "for each session t with >= 11 prior defined
# sessions" -- one more than the ten-value window itself needs.
_MIN_PRIOR_DEFINED_SESSIONS = 11

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


@dataclass(frozen=True)
class BreadthSnapshot:
    """The snapshot as this module consumes it.

    ``dates`` is the snapshot's own defined-session calendar (sorted, unique,
    tz-naive); ``series`` is the lookup-only handle every VALUE is read
    through; ``unknown_sessions`` is the union of the provenance's broken and
    omitted session lists (the only record of holes an as_of consumer has).
    """

    dates: pd.DatetimeIndex
    series: panel.AsOfSeries
    unknown_sessions: frozenset


def _normalize_index(index, label: str) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_localize(None)
    out = out.normalize()
    if len(out) == 0:
        raise ValueError(f"{FAMILY_ID}: {label} is empty")
    if not out.is_monotonic_increasing or out.has_duplicates:
        raise ValueError(f"{FAMILY_ID}: {label} must be sorted and unique")
    return out


def load_snapshot(repo_root: str | None = None) -> BreadthSnapshot:
    """Load the pinned snapshot: sha-verified handle + its calendar + holes."""
    spec = SNAPSHOTS[_SNAPSHOT_NAME]
    path = spec["path"]
    if not os.path.isabs(path):
        path = os.path.join(repo_root or _REPO_ROOT, path)
    series = panel.load_breadth_snapshot(path, expect_sha256=spec["sha256"])
    if series.indicator != spec["indicator"]:
        raise ValueError(
            f"{FAMILY_ID}: snapshot indicator {series.indicator!r} is not the "
            f"declared {spec['indicator']!r}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("content_sha256") != spec["sha256"]:
        raise ValueError(f"{FAMILY_ID}: snapshot sha changed between reads")
    if payload.get("parameters") != spec["params"]:
        raise ValueError(
            f"{FAMILY_ID}: snapshot parameters {payload.get('parameters')!r} are "
            f"not the declared {spec['params']!r}")
    dates = _normalize_index(payload["dates"], "snapshot dates")
    if len(dates) != len(series):
        raise ValueError(
            f"{FAMILY_ID}: snapshot has {len(dates)} dates but the loaded "
            f"series has {len(series)} observations")
    provenance = payload["provenance"]
    unknown = frozenset(
        pd.Timestamp(day).normalize()
        for key in ("broken_sessions", "omitted_sessions")
        for day in provenance.get(key, []))
    return BreadthSnapshot(dates=dates, series=series, unknown_sessions=unknown)


def thrust_dates(snapshot: BreadthSnapshot, last) -> pd.DatetimeIndex:
    """Snapshot sessions dated <= ``last`` on which a Zweig thrust fires.

    Reads every defined session <= ``last`` through ``as_of`` (the causal EMA
    must be seeded at the snapshot's FIRST defined session, so warm-up before
    a bucket is required), computes share -> EMA -> the from-below-0.40 to
    above-0.615-within-10-sessions test, and refuses any recorded unknown
    session inside the consulted span.
    """
    last = pd.Timestamp(last)
    if last.tz is not None:
        last = last.tz_localize(None)
    last = last.normalize()
    dates = snapshot.dates[snapshot.dates <= last]
    if len(dates) == 0:
        return pd.DatetimeIndex([])
    holes = sorted(day for day in snapshot.unknown_sessions
                   if dates[0] <= day <= last)
    if holes:
        shown = [str(day.date()) for day in holes[:25]]
        raise ValueError(
            f"{FAMILY_ID}: refusing to run the EMA across {len(holes)} unknown "
            f"session(s) {shown} recorded in the snapshot's provenance; a "
            "path-dependent breadth statistic may not bridge a hole.")

    values = np.empty(len(dates), dtype=float)
    for position, day in enumerate(dates):
        value = snapshot.series.as_of(day)
        if value is None:
            raise ValueError(f"{FAMILY_ID}: as_of({day.date()}) returned None on a "
                             "defined snapshot session")
        values[position] = value
    if not np.isfinite(values).all():
        raise ValueError(f"{FAMILY_ID}: snapshot values must be finite")

    share = (1.0 + values) / 2.0
    span = int(PARAMETERS["ema_span"])
    alpha = 2.0 / (span + 1.0)
    z = np.empty_like(share)
    z[0] = share[0]
    for k in range(1, len(share)):
        z[k] = alpha * share[k] + (1.0 - alpha) * z[k - 1]

    window = int(PARAMETERS["window_sessions"])
    prior_min = pd.Series(z).rolling(window, min_periods=window).min().shift(1)
    fires = ((z > float(PARAMETERS["trigger_level"]))
             & (prior_min.to_numpy() < float(PARAMETERS["setup_level"])))
    fires[:_MIN_PRIOR_DEFINED_SESSIONS] = False
    return pd.DatetimeIndex(dates[fires])


def hold_schedule(thrusts, index: pd.DatetimeIndex) -> tuple:
    """Replay the pseudocode's state machine over the bucket ``index``.

    Returns ``(entry_flags, exit_flags)`` -- boolean arrays aligned to
    ``index``: entry on the thrust bar t (fills at t+1), exit on bar t+63.
    """
    bucket = _normalize_index(index, "index")
    hold = int(PARAMETERS["hold_sessions"])
    thrust_set = {pd.Timestamp(day).normalize() for day in thrusts}
    entry = np.zeros(len(bucket), dtype=bool)
    exit_ = np.zeros(len(bucket), dtype=bool)
    in_position = False
    exit_position = None
    for t in range(len(bucket)):
        if not in_position and bucket[t] in thrust_set:
            if t + 1 >= len(bucket):
                continue  # open(t+1) does not exist inside this bucket
            entry[t] = True
            in_position = True
            exit_position = t + hold  # t+1 fill plus hold-1 more sessions
        elif in_position and t == exit_position:
            exit_[t] = True
            in_position = False
            exit_position = None
    return entry, exit_


def make_signal_fn(ticker: str, index: pd.DatetimeIndex, frames: dict, *,
                   snapshot: BreadthSnapshot | None = None):
    """Build the per-bar SignalFn for the bucket ``index`` (see module doc).

    ``snapshot`` is injectable for tests only; production loads the pinned
    file through ``load_snapshot``.
    """
    traded = TRADED[0]
    if ticker != traded:
        raise ValueError(f"{FAMILY_ID}: make_signal_fn expects ticker {traded!r}, "
                         f"got {ticker!r}")
    if not isinstance(frames, dict) or traded not in frames:
        raise ValueError(f"{FAMILY_ID}: frames lacks {traded}")
    bucket = _normalize_index(index, "index")
    if snapshot is None:
        snapshot = load_snapshot()

    thrusts = thrust_dates(snapshot, bucket[-1])
    entry_arr, exit_arr = hold_schedule(thrusts, bucket)
    bucket_values = bucket.asi8

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        today = frame.index[-1]
        if today.value != bucket_values[i]:
            raise ValueError(
                f"{FAMILY_ID}: signal_fn bar {i} is {today} but the bucket "
                f"index has {bucket[i]} -- frame/index misaligned")
        return SignalResult(entry=bool(entry_arr[i]), exit=bool(exit_arr[i]))

    return signal_fn
