"""Batch5 candidate: short_rate_easing_smallcap_rotation (rotation_two_asset).

PRE-REGISTERED RULE (both constants fixed BEFORE any P&L existed
for this family and never to be tuned after a backtest result)::

    at each month-end m (last trading session of the month):
        rtb    = irx_close[m] - mean(irx_close over the trailing 252
                 sessions ending at m, m included)     # percentage points
        easing = rtb <= -0.10
        target = IWM if easing else SPY

Mechanism: monetary easing raises stock returns and small firms are the more
credit-sensitive (Thorbecke 1997, Journal of Finance 52(2):635-654;
Perez-Quiros and Timmermann 2000, Journal of Finance 55(3):1229-1262); the
short rate is a return predictor (Ang and Bekaert 2007, Review of Financial
Studies 20(3):651-707); fixed-window detrending of the bill rate follows the
Campbell (1991)/Hodrick (1992) "relative bill rate" tradition. Citation
status: bibliographic details of the three main papers
and their abstract-level findings were verified; that SMALL stocks respond
more to policy in Thorbecke comes from a secondary summary only; the
Campbell/Hodrick detrending tradition was confirmed only through a secondary
source and the window those papers use was NOT verified (UNVERIFIED). None
of the papers tests THIS rule: the 252-session window and the -0.10 threshold
are this pipeline's own pre-registered constants, not literature values.

Conventions, each verified from source or by test:

* Timing. ``mining.rotation_engine.run_two_asset_rotation`` hands the target
  factory frames truncated through decision bar t and fills its answer at
  bar t+1's Open (rotation_engine.py: ``decision_idx = execution_idx - 1``).
  A reading dated at month-end m is therefore held from the FIRST session of
  the next month, at that session's Open.
* Month-end = the exchange's SCHEDULED last session of each calendar month
  (``scheduled_month_end``: the last weekday, stepping back over Good Friday
  and Memorial Day, the only two scheduled closures that can occupy a
  month's last weekday), taken only where that date is present in the
  TRADED calendar (the IWM and SPY frames' shared dates). It is fixed in
  advance and read from no data file, so no later observation -- not even
  the existence of a later session -- can decide it. A scheduled month-end
  absent from the data (a vendor hole, an unscheduled closure) has no
  reading, and the code never falls back to an earlier day, which would need
  foreknowledge of the closure (the bullet on missing readings below says
  what the factory does about such a hole). Checked against the real
  history: the schedule reproduces the last session that actually traded in
  all 404 complete SPY months since 1993 and all 316 IWM months since 2000,
  with zero mismatches.
* rtb uses ^IRX's OWN cleaned session history, straight off the frame handed
  to ``make_target_fn`` (both drivers hand it the full, unaligned frame), so
  the 252-session window is warm from 1960 and there is no warm-up gap after
  IWM's 2000-05-26 inception. Differences only -- never ratios, logs or a
  floor: 7 real ^IRX closes are at or below zero (minimum -0.105 on
  2020-03-26).
* ``easing = rtb <= -0.10`` is inclusive and compared as a plain float, no
  tolerance. The window mean is a direct numpy mean per month-end (not a
  running/online rolling mean), so no accumulated rounding can move a
  decision.
* A traded month-end without a defined reading (no ^IRX print that day, or
  fewer than 252 prints up to and including it) gets NO reading, never a
  guess. Before the first reading that is just the warm-up. AFTER it, a
  complete month with no reading (a vendor hole, or a session missing from
  a traded frame) makes
  ``make_target_fn`` RAISE by default, naming the month
  (``require_complete_month_ends``), because the score it feeds is one-shot
  and permanent and a silently skipped decision must never reach it; the
  opt-out (False) carries the previous regime through such a month, one
  decision skipped. If NO month-end has a reading at all the factory raises
  too (a family that could never trade must not be scored as if it had been
  tested).
* The factory returns ``signals.make_external_regime_target_fn`` over the
  sparse month-end series (the same forward-carry mechanism
  ``defensive_sector_recession_signal_rotation`` relies on). Its first
  argument is named ``risk_on`` there; here the flag is EASING and the
  "risk_ticker" is IWM (the ticker returned when the flag is True), SPY the
  other. The state is computed ONCE from the full frames at build time, from
  data at or before each month-end only, so (a) no decision can see a later
  observation of any kind (prints, prices or calendar; tested by cutting
  every frame off exactly at a regime-changing month-end), (b) a restricted
  bucket that starts mid-regime -- and whose ^IRX slice is far shorter than
  252 sessions -- gets the carried-in regime from its first bar (verified
  through the real engine, not only by comparing the factory with itself).

Some month-end decisions can sit very close to the band edge; that is a
fact about the rule, disclosed, not a reason to move the threshold.

DISCLOSURES (read before believing any score):

1. Very low statistical power. Fed easing cycles are rare -- only a handful
   in a quarter-century -- and short blips that briefly cross the band are
   not independent bets, so the number of genuinely independent easing
   cycles is smaller still.
2. Easing cycles often coincide with bear markets, so the rotation can hold
   IWM THROUGH a bear market: a size-times-policy bet with no crash
   protection. A good score would need to be read against that.
3. Overlap with ``defensive_sector_recession_signal_rotation`` (also
   ^IRX-based). The ABI-1 scoring path applies NO
   pairwise-correlation correction between families -- every family is
   scored at the fixed ``constants.ABI1_N_TRIALS`` / ``DAILY_SR_VARIANCE``
   (``abi1_driver.evaluate_family`` passes both to
   ``account_relative.evaluate``), so this shared-signal-source overlap is
   NOT corrected for; it is a disclosed concern, not a handled one. Also
   overlapping in spirit: ``size_premium_smallcap`` holds small caps against
   the same SPY benchmark all the time; this family is that bet switched on
   only in easing regimes.
4. The account is always invested once the first reading exists (SPY by
   default), so it carries no cash-drag handicap against the SPY benchmark
   that ABI-1 always uses (a family that sits in cash pays that drag in
   every bull market). On a synthetic never-easing path through the real
   ABI-1 curve builder, the account's daily return after the first entry
   matches the cost-adjusted SPY benchmark's to within 1e-12 on every
   session, so before tax the scored excess is the IWM-minus-SPY return
   earned in easing months plus per-switch execution costs (and the few
   sessions before the first reading, when the account is in cash) -- a much
   narrower bet than "small caps versus the market". TAX is on top: the
   catalog lists ``taxable`` first, so ABI-1 applies realization tax
   (``constants.STCG_TAX_RATE`` on gains from positions held under a year,
   ``constants.LTCG_TAX_RATE`` beyond) to the account but never to the SPY
   benchmark, and the final open position is taxed as if closed at the end of
   the data (on synthetic identical IWM and SPY prices with zero trading
   costs, tax alone lowers the account's cumulative log excess), so a
   genuine but small edge can be eaten by it.
5. Known sensitivity, pinned by a test as a limitation and NOT fixed (the
   pre-registered rule has no outlier handling): one wild HIGH print shifts
   the trailing mean by (print - level)/252 and can hold IWM for up to a year
   (a 30 against a flat 2 reads rtb = -0.111); one wild LOW month-end print
   moves the account for exactly one month.
6. Only the ABI-1 driver is supported. The per-trade bucket driver
   ``run_batch3_mining.run_rotation_windows`` normalizes SIGNAL frames with
   the strict positivity gate and rejects ^IRX's genuine negative prints
   (``^IRX: OHLC values must be finite and positive``);
   ``scripts/run_abi1_mining.py`` loads a symbol that no plan in the run
   trades with ``require_positive=False``, which is what lets ^IRX through.
   Do not batch this family with another one that TRADES ^IRX.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dateutil.easter import easter

from mining import signals

FAMILY_ID = "short_rate_easing_smallcap_rotation"
SHAPE = "rotation_two_asset"
TRADED = ["IWM", "SPY"]        # [target while EASING, target otherwise]
SIGNAL = ["^IRX"]              # 13-week T-bill yield index, never traded
BENCHMARK = "SPY"              # ABI-1 always benchmarks SPY regardless
SNAPSHOTS: dict = {}
# Copied byte-for-byte into data/mining/catalog.json's entry for this
# family_id (this module's own test reads that file directly and compares
# canonical JSON, so the two cannot drift silently). Both numbers are the
# pre-registered constants.
PARAMETERS = {
    "short_rate_ticker": "^IRX",
    "detrend_window_sessions": 252,
    "easing_threshold_pp": -0.10,
    "rebalance": "monthly",
}
_WINDOW = int(PARAMETERS["detrend_window_sessions"])
_EASING_THRESHOLD_PP = float(PARAMETERS["easing_threshold_pp"])
# A conservative pooled floor for a monthly two-asset rotation driven by a
# rare regime flag (each regime transition plus the first entry is one
# segment), the same convention as every sibling module.
EXPECTED_MIN_TRADES = 10


def _require_frame(frames, symbol: str) -> pd.DataFrame:
    frame = frames.get(symbol) if isinstance(frames, dict) else None
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{FAMILY_ID}: frames must contain a non-empty "
                         f"DataFrame for {symbol}")
    if "Close" not in frame.columns:
        raise ValueError(f"{FAMILY_ID}: {symbol} frame is missing Close")
    return frame


def _clean_close(frame: pd.DataFrame) -> pd.Series:
    """Tz-naive, sorted, de-duplicated (keep last) Close series of finite
    values. NO positivity requirement and no floor: a bill rate can print at
    or below zero, and those prints are decision inputs like any other."""
    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    close = pd.Series(pd.to_numeric(frame["Close"], errors="coerce").to_numpy(),
                      index=index)
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close[np.isfinite(close.to_numpy())]


def _good_friday(year: int) -> pd.Timestamp:
    return pd.Timestamp(easter(year)) - pd.Timedelta(days=2)


def _memorial_day(year: int) -> pd.Timestamp:
    """The last Monday of May."""
    day = pd.Timestamp(year, 5, 31)
    while day.weekday() != 0:
        day -= pd.Timedelta(days=1)
    return day


def scheduled_month_end(year: int, month: int) -> pd.Timestamp:
    """The last session of the month as the exchange's published calendar
    fixes it IN ADVANCE: the last weekday, stepping back over the only two
    scheduled full-day closures that can land on a month's last weekday
    under the exchange's post-1971 rules (the traded history here starts in
    2000), Good Friday (March) and Memorial Day (when it is May 31). New
    Year's Day, Independence Day, Christmas and Juneteenth never do (and the
    exchange stays open on a Friday December 31 even when January 1 is a
    Saturday). Deliberately
    independent of any data file: whether a date is a month-end is knowable
    before the day, so no later observation can decide it. Checked against
    every complete month of real SPY history since 1993 and IWM since 2000
    (see the module docstring's month-end convention)."""
    day = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
    closed = {_good_friday(year), _memorial_day(year)}
    while day.weekday() >= 5 or day in closed:
        day -= pd.Timedelta(days=1)
    return day


def month_end_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The scheduled month-ends of every month ``calendar`` spans that are
    actually PRESENT in ``calendar``. A scheduled month-end missing from the
    data (a vendor hole or an unscheduled closure) yields no month-end for
    that month -- never an earlier day picked with hindsight."""
    if len(calendar) == 0:
        return calendar[:0]
    months = pd.period_range(calendar.min(), calendar.max(), freq="M")
    scheduled = pd.DatetimeIndex(
        [scheduled_month_end(p.year, p.month) for p in months])
    return calendar[calendar.isin(scheduled)]


def traded_calendar(frames) -> pd.DatetimeIndex:
    """The sessions both traded legs share (the calendar the rotation engine
    trades on once its own alignment has run)."""
    return _clean_close(frames["IWM"]).index.intersection(
        _clean_close(frames["SPY"]).index).sort_values()


def relative_bill_rate_at(irx_close: pd.Series,
                          dates: pd.DatetimeIndex) -> pd.Series:
    """rtb = irx_close[d] - mean(irx_close over the _WINDOW sessions of
    irx_close ending at d, d included), in percentage points, for each
    date in dates. NaN where irx_close has no print on that date or
    fewer than _WINDOW prints up to and including it. A direct numpy mean
    per date (not a running mean), reading only prints at or before d."""
    values = irx_close.to_numpy(dtype=float)
    positions = irx_close.index.get_indexer(dates)
    out = np.full(len(dates), np.nan)
    for k, position in enumerate(positions):
        if position >= _WINDOW - 1:
            window = values[position - _WINDOW + 1:position + 1]
            out[k] = values[position] - window.mean()
    return pd.Series(out, index=dates)


def is_easing(rtb):
    """The pre-registered band: easing when rtb <= the threshold, edge
    INCLUSIVE, plain float comparison (NaN is never easing)."""
    return np.asarray(rtb) <= _EASING_THRESHOLD_PP


def easing_state_series(irx_close: pd.Series,
                        calendar: pd.DatetimeIndex) -> pd.Series:
    """1.0 (easing) / 0.0 (not) at every traded month-end (a scheduled
    month-end present in ``calendar``) whose reading is defined; months
    without a defined reading are simply absent."""
    ends = month_end_dates(calendar)
    rtb = relative_bill_rate_at(irx_close, ends)
    defined = rtb.dropna()
    return pd.Series(is_easing(defined.to_numpy()).astype(float),
                     index=defined.index)


def _months_without_a_reading(easing: pd.Series,
                              calendar: pd.DatetimeIndex) -> list:
    """Scheduled month-ends of COMPLETE months, from the month of the first
    reading onward, that have no reading. A month is complete when its
    scheduled month-end is not after the last date of ``calendar``; the month
    still in progress and the warm-up before the first reading are never
    counted."""
    last = calendar.max()
    have = set(easing.index)
    missing = []
    for period in pd.period_range(easing.index.min(), last, freq="M"):
        month_end = scheduled_month_end(period.year, period.month)
        if month_end <= last and month_end not in have:
            missing.append(month_end)
    return missing


def make_target_fn(frames, *, require_complete_month_ends: bool = True):
    """Rotation target factory (driver contract). ``frames`` are the
    full-history frames for IWM, SPY and ^IRX, handed once.

    ``require_complete_month_ends`` (default True; neither driver passes it)
    makes the factory RAISE, naming the month, when a complete month after the
    first reading has no reading (a vendor hole in the bill-rate file or a
    missing traded session): the score this feeds is one-shot and permanent,
    so a silently skipped decision must never reach it. Passing False builds
    anyway and carries the previous regime through such months (used by this
    module's own tests of that behaviour)."""
    for symbol in (*TRADED, *SIGNAL):
        _require_frame(frames, symbol)
    irx_close = _clean_close(frames["^IRX"])
    calendar = traded_calendar(frames)
    easing = easing_state_series(irx_close, calendar)
    if easing.empty:
        raise ValueError(
            f"{FAMILY_ID}: no month-end has a defined reading (needs at "
            f"least {_WINDOW} ^IRX prints up to and including a traded "
            "month-end); a family that could never trade must not be scored")
    if require_complete_month_ends:
        missing = _months_without_a_reading(easing, calendar)
        if missing:
            shown = ", ".join(d.strftime("%Y-%m") for d in missing[:6])
            raise ValueError(
                f"{FAMILY_ID}: {len(missing)} complete month(s) after the "
                f"first reading have no reading at their scheduled "
                f"month-end (first: {shown}); refetch the data, or pass "
                "require_complete_month_ends=False to carry the previous "
                "regime through them")
    return signals.make_external_regime_target_fn(
        easing, risk_ticker="IWM", defensive_ticker="SPY")
