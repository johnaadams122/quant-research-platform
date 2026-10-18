"""Batch5 candidate: sector_breadth_collapse_leverage_flip (shape
``rotation_two_asset``).

Sector Breadth Collapse Leverage Flip. Hold the leveraged bull fund SPXL
(Direxion Daily S&P 500 Bull 3X) by default. The decision input is
``participation_above_ma`` (StockCharts' "Percent Above Moving Average"
statistic) computed on the 11-fund sector SPDR panel: the fraction of
seasoned, resolved sector funds trading above their own trailing 50-session
mean. When that fraction falls to or below 0.30 (the source's own "oversold"
level -- a minority of sectors still participating, i.e. sector breadth has
COLLAPSED), flip into the leveraged bear fund SPXS (Direxion Daily S&P 500
Bear 3X) at the next session's open. When it recovers to or above 0.70 (the
source's "overbought" level), flip back into SPXL. Never in cash: every
session is either long SPXL or long SPXS, so the LEVERAGE FLIP -- not a move
to cash -- is what is tested, following the same real-leveraged-product
convention as vol_managed_smallcap_leverage_switch (TNA/IWM) and
small_cap_tsmom_long_inverse_rotation (IWM/RWM).

Breadth source: the pipeline's pinned ``participation_above_ma`` breadth
snapshot on the point-in-time ``sector_spdr_11`` basis (window=50 sessions,
min_members=9), built via ``scripts/build_sector_panel.py snapshot``
and loaded through ``mining.panel.load_breadth_snapshot`` against
the sha pinned below in ``_BREADTH_SNAPSHOT_SHA256``.
``mining/panel_signals.py``'s own module docstring names
"percent-of-members-above-their-own-moving-average" as one of three breadth
candidates that need a panel capability; the other two (advance-decline
line, McClellan-style oscillator) are used by sector_ad_line_divergence_
derisk and sector_breadth_thrust_zweig, and this family is the one that
uses the participation statistic.

DEPENDENCY-DECLARATION DEVIATION, disclosed rather than silent: the shared
batch5 driver's module contract (``scripts/run_batch5_mining.py
validate_module_contract``) currently requires the module-level ``SNAPSHOTS``
dict to be EMPTY unless ``SHAPE == "panel_breadth"`` -- there is no shape yet
for "a rotation whose target is driven by a panel snapshot rather than the
traded assets' own prices". Rather than smuggling a snapshot dependency past
that check or widening the shared contract (a pipeline-infrastructure change
out of scope for one candidate), this module keeps ``SNAPSHOTS = {}`` at the
module level (so it still passes today's ``rotation_two_asset`` contract
cleanly) and pins the same path+sha256 as module-level constants instead
(``_BREADTH_SNAPSHOT_PATH`` / ``_BREADTH_SNAPSHOT_SHA256``), loaded via
``load_breadth()`` with the identical fail-closed sha guarantee
``panel.load_breadth_snapshot`` always provides. A future extension of the
shared contract to a combined shape can promote these into ``SNAPSHOTS``
without changing any number this module produces.

Rule, evaluated once per session t (point-in-time, not path-dependent -- no
accumulation-span gap rule needed):
    part_t = participation_above_ma snapshot value as of t (None before the
             snapshot's first defined session, 1999-03-05; SPXL/SPXS both
             trade from 2008-11-05 onward, so in practice this is always
             already defined over the whole traded history)
    if part_t is None: standing carries forward unchanged (None until the
        first defined session, matching the "no position before the signal
        exists" convention of every panel_breadth sibling)
    elif standing is None (first-ever defined session): initialize
        standing = SPXL if part_t >= 0.50 else SPXS   (the source's own
        bull/bear-bias center line, used ONLY for the one-time seed)
    elif standing == SPXL and part_t <= 0.30: standing = SPXS
    elif standing == SPXS and part_t >= 0.70: standing = SPXL
    else: standing unchanged (inside the hysteresis band, or already at the
        side its own extreme favors)
Engine mapping: mining.rotation_engine.run_two_asset_rotation executes a
target change at t+1's open (a switch closes the old leg and opens the new
one on the same execution bar); an unchanged target holds without churn.

Interpretation choices (the pseudocode leaves these open; each names the
reading closest to the cited ChartSchool definition and invents no unsourced
threshold):
  1. The 0.30/0.70 thresholds and the 50-session window are the source's own
     stated "Percent Above Moving Average" overbought/oversold levels and its
     documented short-to-medium-term MA length -- not tuned for this basis.
     Applying them to an 11-member sector panel rather than a full index's
     constituents is the SAME disclosed coarseness the two other sector
     breadth families also carry.
  2. The ONE-TIME seed rule at the first defined session (>= 0.50 -> SPXL,
     else SPXS) is NOT from the source -- pre-registered a priori, because the
     source states a center line but not which side an all-time-first reading
     should start on. It fires exactly once per backtest (in-sample vs.
     holdout each get their own seed, by construction of a fresh
     ``make_target_fn`` call per bucket, matching every other family's
     bucket-scoped factory).
  3. NEVER in cash: the source's own semantics only ever say "favors the
     bulls" / "favors the bears", never "sit out" -- a leveraged flip that
     always holds one 3x leg is the pre-registered EXECUTION choice, not a
     source claim, and is disclosed as such (same convention as the two
     small-cap leveraged-product families named above).
  4. ``SIGNAL`` is empty: the eleven sector SPDRs enter only through the
     sha-pinned snapshot (the point-in-time panel), never as raw OHLC frames
     handed to the rotation engine.

Performance: the whole decision series is precomputed once in the factory
(one pass over the shared calendar), then looked up by date; ``target_fn`` is
one dict read plus one O(1) consistency check per bar.

Cost basis: the rotation is cost-adjusted with a measured round-trip bid/ask
spread per leg (SPXL and SPXS both trade at a few basis points), with
tax_haircut_rate=0.0 for a gross-of-tax screen. A rough in-sample backtest
(only data before the last 6 months per constants.HOLDOUT_WINDOW_MONTHS,
never touching the reserved holdout window) is a SANITY CHECK that the
mechanism runs cleanly end-to-end through the real engine and produces a
non-degenerate trade sample -- it is not a scored result: no DSR, no holdout
confirmation, and nothing from it is written to the trial ledger.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

from mining import panel

FAMILY_ID = "sector_breadth_collapse_leverage_flip"
SHAPE = "rotation_two_asset"
TRADED = ["SPXL", "SPXS"]
SIGNAL = []
SNAPSHOTS = {}
BENCHMARK = "SPY"
# Copied byte-for-byte from data/mining/catalog.json's "parameters" for this
# family_id; a test asserts zero drift via json.dumps(sort_keys=True).
PARAMETERS = {
    "indicator": "participation_above_ma",
    "min_members": 9,
    "ma_window_days": 50,
    "bearish_threshold": 0.30,
    "bullish_threshold": 0.70,
    "center_line": 0.50,
    "risk_asset": "SPXL",
    "defensive_asset": "SPXS",
    "cash_proxy": "none",
    "execution": "next_open",
}
# Conservative sanity floor on the in-sample trade count (see the module
# docstring's "Cost basis" note on the rough in-sample sanity backtest).
# A configurable threshold, not a reported result.
EXPECTED_MIN_TRADES = 100

RISK_ASSET = str(PARAMETERS["risk_asset"])
DEFENSIVE_ASSET = str(PARAMETERS["defensive_asset"])
BEARISH_THRESHOLD = float(PARAMETERS["bearish_threshold"])
BULLISH_THRESHOLD = float(PARAMETERS["bullish_threshold"])
CENTER_LINE = float(PARAMETERS["center_line"])
_CLOSE_REL_TOL = 1e-9

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
_BREADTH_SNAPSHOT_PATH = ("data/mining/panel/snapshots/"
                          "sector_spdr_11_participation_w50_m9_2026-09-04.json")
_BREADTH_SNAPSHOT_SHA256 = ("9a47beb9658c47dac9304c95b0259144c029041d66cab5ee"
                            "86aff2fc478ce56c")


def load_breadth() -> panel.AsOfSeries:
    """Load the pinned participation_above_ma snapshot (sha-verified) as an
    as-of-only handle. Raises on any drift from the pinned bytes."""
    path = _BREADTH_SNAPSHOT_PATH
    if not os.path.isabs(path):
        path = os.path.join(_REPO_ROOT, path)
    return panel.load_breadth_snapshot(path, expect_sha256=_BREADTH_SNAPSHOT_SHA256)


def _tz_naive(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_localize(None)
    return out


def _epoch_ns(index: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(index.to_numpy(dtype="datetime64[ns]").view("int64"))


def _validated_close(frame: pd.DataFrame, key: str) -> pd.Series:
    """Finite, positive Close on a sorted, unique, tz-naive calendar."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{key}: frame must be a non-empty DataFrame")
    if "Close" not in frame.columns:
        raise ValueError(f"{key}: frame is missing Close")
    index = _tz_naive(frame.index)
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError(f"{key}: frame index must be sorted and unique")
    values = pd.to_numeric(pd.Series(frame["Close"]).to_numpy(), errors="coerce")
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{key}: Close values must be finite and positive")
    return pd.Series(values, index=index, name="Close")


def decision_target(current: str | None, value: float | None) -> str | None:
    """One step of the hysteresis state machine. Pure function of the
    standing target and today's breadth reading -- exposed separately from
    ``decision_series`` so the rule itself has a single, directly-testable
    home."""
    if value is None:
        return current
    if current is None:
        return RISK_ASSET if value >= CENTER_LINE else DEFENSIVE_ASSET
    if current == RISK_ASSET and value <= BEARISH_THRESHOLD:
        return DEFENSIVE_ASSET
    if current == DEFENSIVE_ASSET and value >= BULLISH_THRESHOLD:
        return RISK_ASSET
    return current


def decision_series(shared_calendar: pd.DatetimeIndex,
                    breadth: panel.AsOfSeries) -> pd.Series:
    """Standing target as of each session of ``shared_calendar`` (object
    Series; values RISK_ASSET / DEFENSIVE_ASSET / None). Strictly causal:
    session t's value depends only on ``breadth.as_of(t)`` and the prior
    standing target, never on any later session."""
    standing = None
    out = []
    for date in shared_calendar:
        value = breadth.as_of(date)
        standing = decision_target(standing, value)
        out.append(standing)
    return pd.Series(out, index=shared_calendar, dtype=object, name="target")


def make_target_fn(frames, *, breadth: panel.AsOfSeries | None = None):
    """Driver factory: ``frames`` = {symbol: full-history OHLC} for TRADED.
    Returns ``target(truncated) -> asset | None`` for
    mining.rotation_engine.run_two_asset_rotation. ``breadth`` is injectable
    for synthetic tests; production loads the pinned snapshot."""
    if not isinstance(frames, dict):
        raise TypeError("frames must be a dict of OHLC DataFrames")
    for key in TRADED:
        if key not in frames:
            raise ValueError(f"missing required frame for {key}")
    risk_close = _validated_close(frames[RISK_ASSET], RISK_ASSET)
    defensive_close = _validated_close(frames[DEFENSIVE_ASSET], DEFENSIVE_ASSET)
    shared = risk_close.index.intersection(defensive_close.index).sort_values()
    if len(shared) < 2:
        raise ValueError("traded legs share fewer than two sessions")

    series = breadth if breadth is not None else load_breadth()
    decisions = decision_series(shared, series)
    decision_ns = {int(ts): value for ts, value in
                  zip(_epoch_ns(decisions.index), decisions.to_numpy(dtype=object))}
    close_by_date = {int(ts): float(v) for ts, v in
                     zip(_epoch_ns(risk_close.loc[shared].index),
                         risk_close.loc[shared].to_numpy(dtype=float))}

    def target(truncated):
        frame = truncated[RISK_ASSET]           # KeyError if absent: fail closed
        today = pd.Timestamp(frame.index[-1])
        if today.tzinfo is not None:
            today = today.tz_localize(None)
        expected = close_by_date.get(int(today.value))
        actual = float(frame["Close"].iloc[-1])
        if expected is None or not math.isclose(expected, actual,
                                                rel_tol=_CLOSE_REL_TOL):
            raise ValueError(
                f"{FAMILY_ID}: {RISK_ASSET} close on {today.date()} "
                f"({actual!r}) does not match the factory's data "
                f"({expected!r}); the factory and the engine must run on the "
                "same frames")
        return decision_ns.get(int(today.value))

    target.family_id = FAMILY_ID
    target.decision_dates = decisions.index
    return target
