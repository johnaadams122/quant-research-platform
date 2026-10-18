"""Batch5 candidate: vix_term_structure_carry (single_long).

VIX spot/3M term-structure carry timing on SVXY (short-VIX-futures ETF).
Source: Simon, D.P. and Campasano, J. (2014), "The VIX Futures Basis:
Evidence and Trading Strategies," Journal of Derivatives, 21(3), 54-69
(SSRN abstract_id=2094510). Rule, verbatim from the pre-registered
catalog record's mechanism:

    Each day, compute ratio_t = close(^VIX)_t / close(^VIX3M)_t.
    ratio_t <  1.0 (contango)      -> hold SVXY long (harvest roll yield).
    ratio_t >= 1.0 (backwardation) -> flat (cash).
    Re-enter SVXY once the ratio drops back below 1.0. Re-checked and
    re-decided every trading day (parameters.rebalance = "daily") -- unlike
    the batch5 monthly-rebalance candidates, there is no standing-decision
    forward-fill here: the ratio is a level comparable on every bar, so
    "hold" is simply the absence of a backwardation reading, not a value
    persisted from an earlier decision date.

Engine mapping (mining.backtest_engine): an ``entry`` flag on bar t fills
at bar t+1's open, an ``exit`` flag on bar t fills at bar t's close. A
contango reading on bar t (ratio_t < 1.0) raises entry on bar t (fills at
the next session's open -- the earliest the engine can buy); a
backwardation reading on bar t (ratio_t >= 1.0) raises exit on bar t
(fills at bar t's own close, matching "close the SVXY position" same-day).
The engine only reads ``entry`` while flat and only reads ``exit`` while
long, so both flags can be computed independently, per bar, straight from
that bar's own ratio value -- there is no state to track in the signal
function itself; the engine's own position bookkeeping is what makes
"stay long while ratio stays < 1.0" and "stay flat while ratio stays >=
1.0" fall out for free.

Design (the four non-negotiable properties):

* NO LOOK-AHEAD -- entry/exit on bar t are decided using ONLY that bar's
  own ^VIX and ^VIX3M closes (same-day, end-of-day information), which per
  the engine contract is read from a frame slice ending exactly at bar t;
  bar t+1's data is never consulted. SVXY's own prices never enter the
  signal.
* PERFORMANCE -- the returned signal_fn does two numpy array lookups by
  position plus one scalar date check; nothing is recomputed over the
  growing slice.
* HOLDOUT-BLIND -- frames are truncated to <= the last bar of ``index``
  before anything is computed, so a holdout bucket's signals cannot depend
  on data after the bucket; history BEFORE the bucket is irrelevant here
  (unlike the monthly candidates, a daily level comparison needs no
  warm-up lookback bar).
* DIRECTION -- always long SVXY or flat cash; never short.

Interpretation choices (the mechanism prose leaves these to the
implementer):

1. The boundary is explicit in the prose ("below 1.0" vs "1.0 or above"),
   so ratio_t == 1.0 exactly is backwardation (exit / stay flat), matching
   ``contango_threshold: 1`` read as an inclusive floor on the flat side:
   entry uses strict ``<``, exit uses ``>=``. The two conditions partition
   every finite, computable ratio -- there is no third state.
2. "Align by date to the SVXY frame" (the catalog's own implementation
   note) means the decision calendar is the ``index`` bucket handed to
   ``make_signal_fn`` (SVXY's own bars): ^VIX and ^VIX3M closes are looked
   up by date on that calendar via ``reindex``, not intersected first. A
   bucket bar with no matching ^VIX or ^VIX3M bar (a data gap, not
   expected in practice since both are CBOE indices published on every
   NYSE session SVXY also trades) is silent -- entry=False, exit=False --
   which leaves the engine's own position state exactly where it was
   (flat stays flat, long stays long), the same "undefined is not flat by
   rule" convention as this batch's other signal-ticker candidate.
3. A non-finite ^VIX or ^VIX3M close, or a non-positive ^VIX3M close (the
   ratio's denominator), also yields no signal for that bar rather than
   raising -- an occasional data glitch should degrade to "no decision
   today," not halt the whole backtest. This mirrors the same "no signal"
   handling other batch5 modules give non-finite closes.
4. Duplicate dates in either signal series keep the LAST occurrence before
   sorting (the ``keep="last"`` convention already used elsewhere in this
   batch for de-duplicating a fetched OHLC index).
5. The bucket's OWN final bar is a decision bar like any other for the
   EXIT side (the engine still evaluates exit on the last bar it holds a
   position through) but is structurally never consulted for the ENTRY
   side -- ``run_backtest`` never calls the flat-side check with the
   bucket's last position, since entering bar i requires a decision from
   bar i-1 and there is no bar past the last one to enter on. No special
   casing is needed in ``signal_fn`` for this; it is a property of the
   engine's loop, not of this candidate's rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "vix_term_structure_carry"
SHAPE = "single_long"
TRADED = ["SVXY"]
SIGNAL = ["^VIX", "^VIX3M"]
SNAPSHOTS = {}
BENCHMARK = "SVXY"
# Copied byte-for-byte from the pre-registered catalog candidate's
# "parameters" field (driver drift check via json.dumps(sort_keys=True)).
PARAMETERS = {
    "signal": "vix_spot_over_vix3m_ratio",
    "contango_threshold": 1,
    "flat_position": "cash",
    "rebalance": "daily",
}
# SVXY (listed 2011) is younger than ^VIX3M, so the common calendar is
# bounded by SVXY. Configured, conservative floor (not a measured count): a
# daily contango/backwardation switch over a decade-plus is expected to
# trade well more often than this; a run below it points at a data or
# wiring problem.
EXPECTED_MIN_TRADES = 40

_THRESHOLD = float(PARAMETERS["contango_threshold"])


def _naive_index(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_localize(None)
    return out.normalize()


def _require_frame(frames, symbol: str) -> pd.DataFrame:
    frame = frames.get(symbol) if isinstance(frames, dict) else None
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{FAMILY_ID}: frames must contain a non-empty OHLC "
                         f"DataFrame for {symbol}")
    if "Close" not in frame.columns:
        raise ValueError(f"{FAMILY_ID}: {symbol} frame is missing Close")
    return frame


def _clean_close(frame: pd.DataFrame, last_bar: pd.Timestamp) -> pd.Series:
    """Tz-naive, sorted, de-duplicated (keep last) Close series truncated to
    <= ``last_bar`` -- the holdout-blind step, done once per signal ticker."""
    close = pd.Series(frame["Close"].to_numpy(), index=_naive_index(frame.index))
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close[close.index <= last_bar]


def vix_ratio_flags(bucket: pd.DatetimeIndex, vix_close: pd.Series,
                     vix3m_close: pd.Series,
                     threshold: float) -> tuple[pd.Series, pd.Series]:
    """Precompute (entry, exit) boolean Series aligned to ``bucket``.

    entry[t] = ratio_t < threshold (contango); exit[t] = ratio_t >= threshold
    (backwardation). A bar whose ^VIX or ^VIX3M close is missing (not in
    either series), non-finite, or has a non-positive ^VIX3M close is
    silent on both flags (docstring choices 2-3) -- ``bucket``,
    ``vix_close`` and ``vix3m_close`` are each looked up by date via
    ``reindex``, never by position, so a gap in one series never misaligns
    the other.
    """
    vix = pd.to_numeric(vix_close.reindex(bucket), errors="coerce").to_numpy(dtype=float)
    vix3m = pd.to_numeric(vix3m_close.reindex(bucket), errors="coerce").to_numpy(dtype=float)
    computable = np.isfinite(vix) & np.isfinite(vix3m) & (vix3m > 0.0)
    ratio = np.full(len(bucket), np.nan)
    ratio[computable] = vix[computable] / vix3m[computable]
    entry = computable & (ratio < threshold)
    exit_ = computable & (ratio >= threshold)
    return (pd.Series(entry, index=bucket, name="entry"),
            pd.Series(exit_, index=bucket, name="exit"))


def make_signal_fn(ticker: str, index, frames):
    """Factory per the batch5 contract: ``index`` is the bucket the engine
    will run over; ``frames`` holds full-history OHLC for TRADED + SIGNAL."""
    if ticker != TRADED[0]:
        raise ValueError(f"{FAMILY_ID}: make_signal_fn expects ticker "
                         f"{TRADED[0]!r}, got {ticker!r}")
    # Presence check only -- the traded frame's own prices are never used to
    # compute the signal (docstring: "SVXY's own prices never enter...").
    _require_frame(frames, ticker)
    vix_frame = _require_frame(frames, SIGNAL[0])
    vix3m_frame = _require_frame(frames, SIGNAL[1])

    bucket = _naive_index(index)
    if len(bucket) == 0:
        raise ValueError(f"{FAMILY_ID}: index must not be empty")
    if not bucket.is_monotonic_increasing or bucket.has_duplicates:
        raise ValueError(f"{FAMILY_ID}: index must be sorted and unique")
    last_bar = bucket[-1]

    vix_close = _clean_close(vix_frame, last_bar)
    vix3m_close = _clean_close(vix3m_frame, last_bar)

    entry_series, exit_series = vix_ratio_flags(bucket, vix_close, vix3m_close,
                                                 _THRESHOLD)
    entry_arr = entry_series.to_numpy(dtype=bool)
    exit_arr = exit_series.to_numpy(dtype=bool)
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
        return SignalResult(entry=bool(entry_arr[i]), exit=bool(exit_arr[i]))

    return signal_fn
