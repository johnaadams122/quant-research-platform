"""Tests for src/mining/cascade_signals.py -- two day-trading mining
candidates, both variations on the same mechanism: a daily directional bias
(price_structure.daily_bias), confirmed by same-direction structure on the
hourly chart (price_structure.classify_reversal_bar), entered on a 5-minute
fair-value-gap (price_structure.fair_value_gaps) retest-and-reject. LONG-ONLY
scope, matching this catalog's established day-trading convention (e.g.
day_trading_signals.make_orb_5min_atr_stop_scalp_signal_fn's own long-only
disclosure).

Every scenario below is hand-derived against small, sparse, synthetic
1-minute OHLC fixtures.

Fixture-construction convention used throughout this file (deliberately
sparse, not a full RTH minute-by-minute tape): both make_fvg_reversal_
cascade_signal_fn and make_cross_asset_divergence_fvg_cascade_signal_fn
derive DAILY, HOURLY and 5-MINUTE bars from the raw 1-minute `frame` the
engine hands them, using pandas groupby/bucketing keyed purely on each
bar's own timestamp -- never on how many real minutes elapsed between
bars. So a single "landmark" 1-minute bar placed exactly at a bucket's
own start timestamp is sufficient to fully define that bucket's OHLC (it
is the bucket's only row: Open=High=Low=Close all come from that one
bar), and a bucket becomes "complete" the instant any LATER bar in the
frame carries a timestamp at or past that bucket's own end boundary --
regardless of how much real wall-clock time was skipped to get there.
This lets every fixture below use one row per 5-minute (or hourly)
bucket of interest instead of a real ~390-row RTH tape, while still
exercising the exact bucket-completeness/ordering rules the module
implements. Every test is driven through the real `run_intraday_backtest`
engine (never a hand-rolled loop), matching test_day_trading_vwap_
decaying_conviction.py's own established convention for this engine.
"""
import os
import sys

import pandas as pd
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining.intraday_backtest_engine import run_intraday_backtest  # noqa: E402
from mining.cascade_signals import (  # noqa: E402
    make_fvg_reversal_cascade_signal_fn,
    make_cross_asset_divergence_fvg_cascade_signal_fn,
)


TZ = "America/New_York"
DAY_T_MINUS_2 = "2026-03-02"  # Monday
DAY_T_MINUS_1 = "2026-03-03"  # Tuesday
DAY_T = "2026-03-04"          # Wednesday


# ---------------------------------------------------------------------
# Fixture-building helpers
# ---------------------------------------------------------------------

def _ts(date_str, time_str):
    return pd.Timestamp(f"{date_str} {time_str}", tz=TZ)


def _row(o, h, l, c, v=1000.0):
    return {"Open": float(o), "High": float(h), "Low": float(l), "Close": float(c), "Volume": float(v)}


def _frame(rows):
    """rows: list of (timestamp, dict) pairs. Sorted defensively -- callers
    concatenate several days' worth of rows and rely on this to land in
    ascending order regardless of how the pieces were assembled."""
    index = pd.DatetimeIndex([ts for ts, _ in rows])
    data = {col: [r[col] for _, r in rows] for col in ["Open", "High", "Low", "Close", "Volume"]}
    return pd.DataFrame(data, index=index).sort_index()


def _empty_ohlcv_frame():
    """A zero-row, tz-aware OHLCV frame -- same shape/index convention as
    intraday_data.fetch_intraday_bars, just empty. Used to prove the
    permissive missing-secondary-data default (scenario 7)."""
    idx = pd.DatetimeIndex([], tz=TZ)
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in ["Open", "High", "Low", "Close", "Volume"]}, index=idx)


def _bullish_bias_rows(day_minus2=DAY_T_MINUS_2, day_minus1=DAY_T_MINUS_1):
    """T-2 daily aggregate (Open=first,High=max,Low=min,Close=last over its
    2 rows): Open=100, High=102, Low=98, Close=101 -- sets a range.

    T-1 daily aggregate: Open=99, High=100, Low=97, Close=99. Low(97) <
    T-2's Low(98) [swept] AND Close(99) > T-2's Low(98) [reclaimed] AND
    High(100) <= T-2's High(102) [NOT an outside bar] -> classify_
    reversal_bar(T-1 vs T-2) = BULLISH_REVERSAL -> daily_bias relabels
    this "bullish". This is the single completed-day comparison (T-1 vs
    T-2) the BIAS step reads as TODAY's (T's) daily bias, and T-1's own
    daily High (100) is the "prior_day_high" the ENTRY target formula
    uses.
    """
    return [
        (_ts(day_minus2, "09:30"), _row(100, 100, 98, 99)),
        (_ts(day_minus2, "09:31"), _row(99, 102, 99, 101)),
        (_ts(day_minus1, "09:30"), _row(99, 99, 97, 97.5)),
        (_ts(day_minus1, "09:31"), _row(97.5, 100, 97.5, 99)),
    ]


def _no_bias_rows(day_minus2=DAY_T_MINUS_2, day_minus1=DAY_T_MINUS_1):
    """Same T-2 as _bullish_bias_rows (range [98, 102]), but T-1's daily
    aggregate (Open=100, High=101, Low=99, Close=100.5) stays FULLY
    INSIDE that range -- no sweep of either extreme, no clean break
    either way -- classify_reversal_bar's "none" case, so daily_bias is
    "none" and the BIAS step must resolve to phase="no_bias" for T.
    """
    return [
        (_ts(day_minus2, "09:30"), _row(100, 100, 98, 99)),
        (_ts(day_minus2, "09:31"), _row(99, 102, 99, 101)),
        (_ts(day_minus1, "09:30"), _row(100, 100, 99, 100)),
        (_ts(day_minus1, "09:31"), _row(100, 101, 99.5, 100.5)),
    ]


def _happy_path_today_rows(day_t=DAY_T):
    """TODAY's (T's) full 1-minute landmark sequence for a complete
    Build-1 trade, one landmark bar per bucket boundary of interest (see
    module docstring for why one bar fully defines a bucket):

      09:30 -> hourly bucket1 [09:30,10:30): O100 H101 L99 C100.
      10:30 -> completes bucket1; defines bucket2 [10:30,11:30):
               O100.5 H102.5 L100 C102.5 (its own Close 102.5 will be >
               bucket1's High 101 once bucket2 itself later completes).
      11:30 -> completes bucket2 -> 2 complete hourly buckets now exist;
               classify_reversal_bar(bucket2 vs bucket1): swept_low
               False (100 > 99), swept_high True (102.5 > 101) but not
               "outside" (needs BOTH swept), Close(102.5) > prev_high
               (101) -> BULLISH_BREAK -> HOURLY CONFIRMS here,
               confirmed_hour_end = 11:30. This same bar also defines the
               first post-confirmation 5-min bucket, P=[11:30,11:35):
               O100 H100.2 L99.8 C100 (a quiet, narrow-range candle).
      11:35 -> completes P; defines the displacement candle
               M=[11:35,11:40): O101 H103 L100.9 C102.5.
      11:40 -> completes M; defines the FVG's successor candle
               S=[11:40,11:45): O102.6 H103.5 L102.4 C103.
               High[P]=100.2 < Low[S]=102.4 -> a bullish fair value gap
               forms at M's own timestamp (11:35), span (100.2, 102.4)
               -- qualifies since 11:35 >= confirmed_hour_end (11:30).
      11:45 -> completes S (the gap is now detectable for the first
               time); defines the retest/reject candle
               R=[11:45,11:50): O103 H103.2 L102 C102.8. Low(102) <=
               gap_high(102.4) [traded into the gap zone] and versus its
               own predecessor S: swept S's Low (102 < 102.4) and closed
               back above it (102.8 > 102.4), not an outside bar (High
               103.2 does not exceed S's High 103.5) -> BULLISH_REVERSAL.
      11:50 -> completes R -> ENTRY TRIGGER fires on THIS call
               (state["stop_price"] = R's own Low = 102). This bar's own
               OHLC is otherwise just a time-marker.
      11:51 -> the engine's own next-bar-open fill: Open=102.9 is the
               real entry_px. R_dist = 102.9-102 = 0.9; 2R = 1.8;
               prior_day_high(100) - entry_px(102.9) = -2.9 (negative,
               since price already broke well above yesterday's high) ->
               target = 102.9 + max(1.8, -2.9) = 104.7. This bar's own
               High/Low (103.0/102.8) touch neither stop nor target.
      12:00 -> High(105.0) >= target(104.7) -> TARGET exit, filled at
               this SAME bar's Close (engine's same-bar-close exit
               convention) = 104.8.
    """
    return [
        (_ts(day_t, "09:30"), _row(100, 101, 99, 100)),
        (_ts(day_t, "10:30"), _row(100.5, 102.5, 100, 102.5)),
        (_ts(day_t, "11:30"), _row(100, 100.2, 99.8, 100)),
        (_ts(day_t, "11:35"), _row(101, 103, 100.9, 102.5)),
        (_ts(day_t, "11:40"), _row(102.6, 103.5, 102.4, 103)),
        (_ts(day_t, "11:45"), _row(103, 103.2, 102, 102.8)),
        (_ts(day_t, "11:50"), _row(102.85, 102.9, 102.75, 102.85)),
        (_ts(day_t, "11:51"), _row(102.9, 103.0, 102.8, 102.95)),
        (_ts(day_t, "12:00"), _row(103.0, 105.0, 102.85, 104.8)),
    ]


def _flat_hourly_never_confirms_rows(day_t=DAY_T):
    """Four consecutive hourly buckets built from four IDENTICAL landmark
    bars (09:30, 10:30, 11:30, 12:30) plus one more marker (13:30) to
    complete the fourth -- classify_reversal_bar reads "none" every
    single time a new bucket completes (an identical predecessor is
    never swept and never cleanly broken), so the hourly-confirmation
    step never sees a bullish_reversal/bullish_break and phase never
    leaves "awaiting_confirmation", despite a real bullish daily bias
    and four real completed hourly buckets checked along the way.
    """
    bar = _row(100, 100.5, 99.5, 100)
    return [
        (_ts(day_t, "09:30"), dict(bar)),
        (_ts(day_t, "10:30"), dict(bar)),
        (_ts(day_t, "11:30"), dict(bar)),
        (_ts(day_t, "12:30"), dict(bar)),
        (_ts(day_t, "13:30"), dict(bar)),
    ]


def _invalidated_then_fresh_gap_today_rows(day_t=DAY_T):
    """Same hourly confirmation as _happy_path_today_rows (bucket1@09:30,
    bucket2@10:30, confirming at 11:30 -> confirmed_hour_end=11:30), then
    TWO fair value gaps in sequence:

    Gap #1 (P1@11:30 / M1@11:35 / S1@11:40, identical values to
    _happy_path_today_rows' P/M/S) forms at M1's timestamp (11:35), span
    (100.2, 102.4) -- exactly as in the happy path. But BEFORE any bar
    retests and rejects it, X@11:45 (Low=95) undercuts poi_low (100.2)
    once its own bucket completes at 11:50's arrival -> gap #1 is
    INVALIDATED and dropped.

    Z@11:50 (Low=94.5, Close=98) is a deliberate trap: versus its own
    predecessor X (Low=95), it swept X's low and closed back above it --
    a textbook BULLISH_REVERSAL, and its Low (94.5) is far inside gap
    #1's old zone (<= 102.4). A buggy implementation that keeps treating
    gap #1 as "the most recent qualifying gap" after invalidation (rather
    than excluding it) would wrongly fire an entry here. A correct one
    finds ZERO qualifying gaps at this point (gap #1 excluded, gap #2 not
    yet formed) and does nothing.

    Gap #2 (P2@11:55 / M2@12:00 / S2@12:05) is a genuinely FRESH gap:
    High[P2]=99.5 < Low[S2]=102.8 -> forms at M2's timestamp (12:00),
    span (99.5, 102.8), detected once S2 completes at 12:10's arrival.
    (M2's own Low is deliberately set to 98.5, BELOW P2's High of 99.5,
    so the P2/Z/M2 triple does NOT also form an accidental extra gap at
    P2's own timestamp -- this fixture wants exactly one live gap at a
    time.)

    W@12:10 (Low=102.5, Close=103.3) retests into gap #2's zone (Low <=
    102.8) and versus its own predecessor S2 (Low=102.8): swept S2's low
    and closed back above it -> BULLISH_REVERSAL -> once W's own bucket
    completes at 12:15's arrival, this is the real, sole entry trigger
    (state["stop_price"] = W's own Low = 102.5).

    12:16 is the engine's real fill bar (Open=103.35). R_dist =
    103.35-102.5 = 0.85; 2R = 1.7; prior_day_high(100) - entry_px
    (103.35) = -3.35 (negative) -> target = 103.35 + 1.7 = 105.05.
    12:20's High (105.5) clears it -> TARGET exit at that bar's Close
    (105.2).
    """
    return [
        (_ts(day_t, "09:30"), _row(100, 101, 99, 100)),           # B0 (hourly bucket1 + 5min B0)
        (_ts(day_t, "10:30"), _row(100.5, 102.5, 100, 102.5)),    # B1 (hourly bucket2 + 5min B1) -> confirms
        (_ts(day_t, "11:30"), _row(100, 100.2, 99.8, 100)),       # P1 (gap #1 predecessor)
        (_ts(day_t, "11:35"), _row(101, 103, 100.9, 102.5)),      # M1 (gap #1 middle -- ts=11:35)
        (_ts(day_t, "11:40"), _row(102.6, 103.5, 102.4, 103)),    # S1 (gap #1 successor)
        (_ts(day_t, "11:45"), _row(102, 102.5, 95, 101)),         # X  -- undercuts gap #1's poi_low(100.2) once complete
        (_ts(day_t, "11:50"), _row(94.8, 99, 94.5, 98)),          # Z  -- trap: textbook bullish_reversal vs a DEAD gap
        (_ts(day_t, "11:55"), _row(99, 99.5, 98.5, 99.2)),        # P2 (gap #2 predecessor)
        (_ts(day_t, "12:00"), _row(100, 103, 98.5, 102.5)),       # M2 (gap #2 middle -- ts=12:00)
        (_ts(day_t, "12:05"), _row(103, 104, 102.8, 103.5)),      # S2 (gap #2 successor)
        (_ts(day_t, "12:10"), _row(103.4, 103.6, 102.5, 103.3)),  # W  -- real retest/reject vs S2
        (_ts(day_t, "12:15"), _row(103.3, 103.4, 103.2, 103.3)),  # marker: completes W -> ENTRY TRIGGER
        (_ts(day_t, "12:16"), _row(103.35, 103.5, 103.2, 103.4)), # fill bar: entry_px=103.35
        (_ts(day_t, "12:20"), _row(103.4, 105.5, 103.1, 105.2)),  # TARGET exit, Close=105.2
    ]


def _second_setup_after_close_rows(day_t=DAY_T):
    """A second, structurally valid-looking FVG-retest setup built ENTIRELY
    after _happy_path_today_rows' own trade has already exited (12:00) --
    P3@12:05/M3@12:10/S3@12:15 form a fresh bullish gap (High[P3]=105 <
    Low[S3]=107.4, at M3's ts=12:10), and W3@12:20 (Low=107.0, Close=
    107.8) retests it and closes back above S3's own Low (107.4) --
    another textbook BULLISH_REVERSAL that WOULD be a second real entry
    trigger if the one-trade-per-day rule were not enforced. The marker
    at 12:25 completes W3's bucket, which is where that (never-taken)
    second entry would fire.
    """
    return [
        (_ts(day_t, "12:05"), _row(104.8, 105, 104.6, 104.9)),   # P3
        (_ts(day_t, "12:10"), _row(105, 108, 104.9, 107.5)),     # M3 (ts=12:10, gap middle)
        (_ts(day_t, "12:15"), _row(107.6, 108.5, 107.4, 108)),   # S3 -- gap forms here
        (_ts(day_t, "12:20"), _row(107.9, 108, 107.0, 107.8)),   # W3 -- would-be retest/reject
        (_ts(day_t, "12:25"), _row(107.8, 107.9, 107.7, 107.8)), # marker: completes W3
    ]


def _secondary_confirms_same_move_rows(day_t=DAY_T):
    """Secondary instrument's own hourly bucket [10:30,11:30) mirrors the
    PRIMARY's bullish_break at the identical hour (Close(52.5) >
    prev_high(51)) -- both instruments move together, so the divergence
    gate must REJECT the primary's own hourly confirmation.
    """
    return [
        (_ts(day_t, "09:30"), _row(50, 51, 49, 50)),
        (_ts(day_t, "10:30"), _row(50.2, 52.5, 50, 52.5)),
        (_ts(day_t, "11:30"), _row(52.5, 52.6, 52.4, 52.5)),  # marker: completes the 10:30 bucket
    ]


def _secondary_diverges_rows(day_t=DAY_T):
    """Secondary instrument's matching hourly bucket [10:30,11:30) is flat
    ("none" vs its own 09:30 bucket) while the primary breaks out -- a
    genuine divergence -- so the gate must ALLOW the primary's own hourly
    confirmation to proceed.
    """
    return [
        (_ts(day_t, "09:30"), _row(50, 51, 49, 50)),
        (_ts(day_t, "10:30"), _row(50, 50.5, 49.5, 50)),
        (_ts(day_t, "11:30"), _row(50, 50.1, 49.9, 50)),  # marker: completes the 10:30 bucket
    ]


# ---------------------------------------------------------------------
# Scenario 1: full happy-path trade (Build 1)
# ---------------------------------------------------------------------

def test_scenario1_happy_path_full_trade_via_target_exit():
    df = _frame(_bullish_bias_rows() + _happy_path_today_rows())
    signal_fn = make_fvg_reversal_cascade_signal_fn()
    trades, daily = run_intraday_backtest(df, signal_fn)

    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["entry_px"] == pytest.approx(102.9)
    assert row["exit_px"] == pytest.approx(104.8)
    assert row["reason"] == "signal_exit"
    entry_ts = row["entry_datetime"]
    exit_ts = row["exit_datetime"]
    assert (entry_ts.hour, entry_ts.minute) == (11, 51)
    assert (exit_ts.hour, exit_ts.minute) == (12, 0)
    assert not daily.empty


# ---------------------------------------------------------------------
# Scenario 2: no-bias day -- no entry ever fires
# ---------------------------------------------------------------------

def test_scenario2_no_bias_day_never_enters():
    # Same T-session bars as the happy path (a fully-formed hourly
    # confirmation + FVG + retest/reject IS present) -- only the prior
    # two days differ (T-1 stays inside T-2's range -> daily_bias="none").
    # Proves the BIAS gate itself blocks entry, not merely that no
    # qualifying setup was built.
    df = _frame(_no_bias_rows() + _happy_path_today_rows())
    signal_fn = make_fvg_reversal_cascade_signal_fn()
    trades, _ = run_intraday_backtest(df, signal_fn)

    assert len(trades) == 0


# ---------------------------------------------------------------------
# Scenario 3: bias exists, but hourly confirmation never completes
# ---------------------------------------------------------------------

def test_scenario3_bias_present_but_hourly_confirmation_never_completes():
    df = _frame(_bullish_bias_rows() + _flat_hourly_never_confirms_rows())
    signal_fn = make_fvg_reversal_cascade_signal_fn()
    trades, _ = run_intraday_backtest(df, signal_fn)

    assert len(trades) == 0


# ---------------------------------------------------------------------
# Scenario 4: an invalidated gap is dropped; a later, FRESH gap still
# triggers entry.
# ---------------------------------------------------------------------

def test_scenario4_invalidated_gap_dropped_then_fresh_gap_triggers_entry():
    df = _frame(_bullish_bias_rows() + _invalidated_then_fresh_gap_today_rows())
    signal_fn = make_fvg_reversal_cascade_signal_fn()
    trades, _ = run_intraday_backtest(df, signal_fn)

    # Exactly one trade all day -- proves the Z@11:50 trap (a textbook
    # bullish_reversal retesting into the now-DEAD gap #1's zone) never
    # produced a trade; only the genuinely fresh gap #2 did.
    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["entry_px"] == pytest.approx(103.35)
    entry_ts = row["entry_datetime"]
    assert (entry_ts.hour, entry_ts.minute) == (12, 16)
    assert row["exit_px"] == pytest.approx(105.2)
    assert row["reason"] == "signal_exit"


# ---------------------------------------------------------------------
# Scenario 5: only one trade fires per day, even if a second qualifying
# setup appears later the same day after the first trade closes.
# ---------------------------------------------------------------------

def test_scenario5_only_one_trade_fires_per_day():
    rows = _bullish_bias_rows() + _happy_path_today_rows() + _second_setup_after_close_rows()
    df = _frame(rows)
    signal_fn = make_fvg_reversal_cascade_signal_fn()
    trades, _ = run_intraday_backtest(df, signal_fn)

    # Still exactly one trade -- the SAME first trade, not a second one
    # off the post-close setup (phase="done_today" gates everything but
    # the exit check, and the first position is already flat).
    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["entry_px"] == pytest.approx(102.9)
    assert row["exit_px"] == pytest.approx(104.8)


# ---------------------------------------------------------------------
# Scenario 5b: a qualifying setup completing AT OR AFTER force_flat_et
# must never fire an entry. Guards step 3's gate
# `current_ts.time() < force_flat_time`: an ungated entry this late could
# fill on the FOLLOWING day, and the unconditional per-day state reset at
# the top of the closure would then wipe `armed` before step 4 ever
# captured that fill -- the position would become invisible to this
# closure's own exit logic and ride an extra full session past the
# same-day-flat invariant before the engine's own end-of-day backstop
# eventually closed it.
# ---------------------------------------------------------------------

def _late_day_setup_rows(day_t=DAY_T):
    """Same shape as _happy_path_today_rows' hourly-confirmation-then-FVG
    cascade, shifted so the retest/reject bucket does not COMPLETE until
    exactly 15:55 -- the default force_flat_et. The bucket-completeness
    rule cares only about landmark placement, not elapsed real time, so
    the confirming bucket [10:30,11:30) is still processed (and completed)
    at the 15:30 landmark -- just later in wall-clock terms than the
    happy path's own 11:30 confirmation.

      09:30 -> hourly bucket0: O100 H101 L99 C100.
      10:30 -> hourly bucket1: O100.5 H102.5 L100 C102.5 (its own row).
      15:30 -> completes bucket1 (end=11:30 <= 15:30) -> HOURLY CONFIRMS
               here (bullish_break, Close(102.5) > bucket0's High(101)),
               confirmed_hour_end=11:30. This SAME row also defines the
               first post-confirmation 5-min bucket P=[15:30,15:35)
               (15:30 is itself a clean 5-min-and-60-min boundary): O100
               H100.2 L99.8 C100.
      15:35 -> completes P; defines M=[15:35,15:40): O101 H103 L100.9
               C102.5.
      15:40 -> completes M; defines S=[15:40,15:45): O102.6 H103.5
               L102.4 C103 -- High[P](100.2) < Low[S](102.4) -> a bullish
               gap forms at M's own ts (15:35), span (100.2, 102.4),
               qualifies since 15:35 >= confirmed_hour_end (11:30).
      15:50 -> completes S (gap now live); defines the retest/reject
               candle R=[15:50,15:55): O103 H103.2 L102 C102.8 -- versus
               S: swept S's Low (102 < 102.4), closed back above it
               (102.8 > 102.4), not an outside bar -> would classify
               BULLISH_REVERSAL once R's own bucket completes.
      15:55 -> completes R (end=15:55 <= 15:55) -- THIS is the exact bar
               where an entry step without the time gate would fire
               entry=True (R's Low(102) <= gap_high(102.4) and
               BULLISH_REVERSAL). current_ts.time() at this call is
               exactly 15:55 == force_flat_time, which step 3's strict `<`
               comparison excludes -- no entry.
      16:00 -> one more marker bar, well past force_flat_et, proving the
               setup stays dead for the rest of the session (not merely
               "not yet triggered").
    """
    return [
        (_ts(day_t, "09:30"), _row(100, 101, 99, 100)),
        (_ts(day_t, "10:30"), _row(100.5, 102.5, 100, 102.5)),
        (_ts(day_t, "15:30"), _row(100, 100.2, 99.8, 100)),
        (_ts(day_t, "15:35"), _row(101, 103, 100.9, 102.5)),
        (_ts(day_t, "15:40"), _row(102.6, 103.5, 102.4, 103)),
        (_ts(day_t, "15:50"), _row(103, 103.2, 102, 102.8)),
        (_ts(day_t, "15:55"), _row(102.85, 102.9, 102.75, 102.85)),
        (_ts(day_t, "16:00"), _row(102.9, 103.0, 102.8, 102.95)),
    ]


def test_scenario5b_setup_completing_at_or_after_force_flat_never_enters():
    df = _frame(_bullish_bias_rows() + _late_day_setup_rows())
    signal_fn = make_fvg_reversal_cascade_signal_fn()
    trades, _ = run_intraday_backtest(df, signal_fn)

    # Zero trades: the retest/reject bucket only COMPLETES at 15:55, at or
    # after the default force_flat_et -- entry must never fire here, and
    # (per step 3's time gate) never silently fires-then-vanishes into a
    # lost next-day fill either. run_intraday_backtest's own end-of-day
    # force-close would surface a phantom position as a trade in `trades`
    # if one had wrongly been opened -- so a clean zero-trade result here
    # is a genuine end-to-end proof, not just an absence of a positive
    # assertion.
    assert len(trades) == 0


# ---------------------------------------------------------------------
# Scenario 6 (Build 2): cross-asset divergence gate.
# ---------------------------------------------------------------------

def test_scenario6a_divergence_gate_blocks_confirmation_when_secondary_confirms_too():
    primary = _frame(_bullish_bias_rows() + _happy_path_today_rows())
    secondary = _frame(_secondary_confirms_same_move_rows())
    signal_fn = make_cross_asset_divergence_fvg_cascade_signal_fn(secondary_bars=secondary)
    trades, _ = run_intraday_backtest(primary, signal_fn)

    # The secondary instrument ALSO shows a bullish_break at the
    # identical hour -- both moving together is not a divergence -- so
    # confirmation must be rejected and no entry can ever fire, despite
    # the primary's own FVG/retest setup being fully present.
    assert len(trades) == 0


def test_scenario6b_divergence_gate_allows_confirmation_when_secondary_diverges():
    primary = _frame(_bullish_bias_rows() + _happy_path_today_rows())
    secondary = _frame(_secondary_diverges_rows())
    signal_fn = make_cross_asset_divergence_fvg_cascade_signal_fn(secondary_bars=secondary)
    trades, _ = run_intraday_backtest(primary, signal_fn)

    # Secondary is flat ("none") at the same hour -- a genuine divergence
    # -- so confirmation proceeds and the trade plays out identically to
    # the Build-1 happy path (scenario 1).
    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["entry_px"] == pytest.approx(102.9)
    assert row["exit_px"] == pytest.approx(104.8)
    assert row["reason"] == "signal_exit"


# ---------------------------------------------------------------------
# Scenario 7 (Build 2): secondary_bars not covering the relevant
# timestamp -- permissive default lets the primary confirm alone.
# ---------------------------------------------------------------------

def test_scenario7_missing_secondary_data_uses_permissive_default():
    primary = _frame(_bullish_bias_rows() + _happy_path_today_rows())
    secondary = _empty_ohlcv_frame()
    signal_fn = make_cross_asset_divergence_fvg_cascade_signal_fn(secondary_bars=secondary)
    trades, _ = run_intraday_backtest(primary, signal_fn)

    # No secondary data at all -> "does not confirm" by the disclosed
    # permissive default -> primary's own confirmation proceeds alone,
    # same outcome as the Build-1 happy path.
    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["entry_px"] == pytest.approx(102.9)
    assert row["exit_px"] == pytest.approx(104.8)


# ---------------------------------------------------------------------
# Scenario 8: insufficient-history early bars never raise, for both
# builds.
# ---------------------------------------------------------------------

def test_scenario8a_no_prior_days_never_raises():
    # Zero completed daily bars (T is the only day present) -> "fewer
    # than 2 completed daily bars" -> phase="no_bias" all day.
    rows = [
        (_ts(DAY_T, "09:30"), _row(100, 101, 99, 100)),
        (_ts(DAY_T, "09:31"), _row(100, 101, 99, 100)),
        (_ts(DAY_T, "09:32"), _row(100, 101, 99, 100)),
    ]
    df = _frame(rows)
    signal_fn = make_fvg_reversal_cascade_signal_fn()
    trades, _ = run_intraday_backtest(df, signal_fn)  # must not raise
    assert len(trades) == 0


def test_scenario8b_hourly_confirmation_insufficient_history_never_raises():
    # A real bullish bias (2 completed daily bars) but only a SINGLE bar
    # for T -> zero complete hourly buckets ever -> "fewer than 2
    # complete hourly buckets" -> stays in awaiting_confirmation, never
    # raises.
    rows = _bullish_bias_rows() + [(_ts(DAY_T, "09:30"), _row(100, 101, 99, 100))]
    df = _frame(rows)
    signal_fn = make_fvg_reversal_cascade_signal_fn()
    trades, _ = run_intraday_backtest(df, signal_fn)  # must not raise
    assert len(trades) == 0


def test_scenario8c_five_minute_insufficient_history_never_raises():
    # Bias present, and hourly confirmation genuinely completes right at
    # the 11:30 bar (2 complete hourly buckets, bullish_break) -- but
    # only 2 complete 5-minute bars exist by then (B0@09:30, B1@10:30),
    # one short of fair_value_gaps' own 3-bar minimum. Must stay in
    # awaiting_entry and never raise.
    rows = _bullish_bias_rows() + [
        (_ts(DAY_T, "09:30"), _row(100, 101, 99, 100)),
        (_ts(DAY_T, "10:30"), _row(100.5, 102.5, 100, 102.5)),
        (_ts(DAY_T, "11:30"), _row(100, 100.2, 99.8, 100)),
    ]
    df = _frame(rows)
    signal_fn = make_fvg_reversal_cascade_signal_fn()
    trades, _ = run_intraday_backtest(df, signal_fn)  # must not raise
    assert len(trades) == 0


def test_scenario8d_build2_never_raises_with_empty_secondary_and_sparse_primary():
    rows = [
        (_ts(DAY_T, "09:30"), _row(100, 101, 99, 100)),
        (_ts(DAY_T, "09:31"), _row(100, 101, 99, 100)),
    ]
    df = _frame(rows)
    secondary = _empty_ohlcv_frame()
    signal_fn = make_cross_asset_divergence_fvg_cascade_signal_fn(secondary_bars=secondary)
    trades, _ = run_intraday_backtest(df, signal_fn)  # must not raise
    assert len(trades) == 0
