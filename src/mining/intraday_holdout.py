"""Holdout-boundary reconciliation for the intraday/swing and day-trading
mining tracks.

## Relationship to scripts/run_batch3_mining.py's own reconciliation

That script's `_mark_to_market_spanning_trades` exists because
`rotation_engine.py`'s trade-level output has NO per-day marks: a rotation
trade only carries its own entry/exit price, so a trade that opened during the
lookback-padding period and is still open when the real holdout window begins
has to be synthetically RE-PRICED (as if freshly opened at the first real
holdout session's open) before it can be counted as real holdout activity.

`intraday_backtest_engine.run_intraday_backtest` does NOT have this
gap for its `daily_returns` output: every touched day -- including the entry
day and every day a position is held through, whether padding or real
holdout -- already gets its own real, honest mark-to-market return (see that
module's docstring, "The mark-to-market design"), computed from that day's
own OHLC marks with no knowledge of any holdout boundary needed. Reconciling
`daily_returns` down to the real holdout window really is just a SLICE -- no
re-pricing, no synthetic entry. See
`test_mining_intraday_holdout.py`'s
`test_position_opened_during_padding_and_held_into_holdout_gets_genuine_marks`.

**`trades_df` is a DIFFERENT story**: `net_ret`/`gross_ret` are
TRADE-LEVEL aggregates, computed ONCE from the trade's original `entry_px`
through `exit_px` -- unlike `daily_returns`, there is no per-day
decomposition to fall back on. A trade that opened during padding and exits
inside real holdout has a `net_ret` that reflects the ENTIRE move, including
however much of it happened before `holdout_start` even began. Simply
keeping that trade's original `net_ret` after slicing would silently leak
pre-holdout performance into `stats_holdout`/`holdout_net_edge_bps`, the
exact number the holdout pass/fail check depends on being genuinely
out-of-sample. `reconcile_
holdout_window` below therefore DOES need essentially the same re-pricing
trick as `_mark_to_market_spanning_trades` for a spanning trade's `net_ret`/
`gross_ret` -- see that function's own docstring for full details, mirrored
here using this engine's own trade-level cost/tax formula. The `daily_
returns` slice needs no such correction; only the trade-level side does.

A trade whose `exit_datetime` lands exactly on `holdout_start` is treated as
INSIDE the holdout window here (unlike batch3's own convention of dropping
it as pure warm-up) -- see `reconcile_holdout_window`'s own docstring for why
this module's day-level granularity makes that safe even after re-pricing.

## Trade-stats and pooling helpers

`pooling.py`'s `pool_trades`/`pooled_trade_stats` (used elsewhere in this
pipeline for the daily engine's pooled multi-ticker runs) are NOT reused
directly here, for two reasons:

- `pool_trades` hardcodes `sort_values(["entry_date", "ticker"])` and an
  empty-frame fallback column list of `["entry_date", "exit_date", ...,
  "held", ...]` -- `backtest_engine.run_backtest`'s own trades_df column
  names. `run_intraday_backtest`'s trades_df uses `entry_datetime` /
  `exit_datetime` / `held_days` instead (see that module's own docstring) --
  `pool_trades` would raise `KeyError: 'entry_date'` on a non-empty call.
- `pooled_trade_stats` itself only ever touches `pooled_trades_df["ticker"]`
  and `["net_ret"]` (column-name-agnostic beyond those two), so it WOULD
  technically run without raising on this engine's trades_df shape -- but it
  omits the `kurtosis >= 1 + skew**2` floor that `dsr.py`'s own docstring
  (`probabilistic_sharpe_ratio`) proves is a hard mathematical necessity for
  `dsr.probabilistic_sharpe_ratio` to not raise `ValueError` on real pooled
  trade data (`_score_one` calls `dsr.deflated_sharpe_ratio`, which calls
  `probabilistic_sharpe_ratio`, downstream of every `stats_in_sample`/
  `stats_holdout` this module produces). `scripts/run_batch3_mining.py`'s own
  private `_trade_stats` DOES carry this floor -- `_trade_stats`
  itself is column-name-agnostic too (only ever touches `trades["net_ret"]`),
  so `intraday_trade_stats` below mirrors ITS formula exactly (including the
  floor), not `pooled_trade_stats`'s.

Additive only -- never imports/modifies `pooling.py`, `backtest_engine.py`,
`run_batch3_mining.py`, or the other intraday-track modules
(`intraday_data.py`, `intraday_backtest_engine.py`, `intraday_cost_gate.py`).
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import scipy.stats

# Calendar-day buffer prepended to the real `holdout_start` when fetching a
# padded window of intraday minute-bar data for a swing candidate's holdout
# evaluation. Unlike scripts/run_batch3_mining.py's own
# HOLDOUT_LOOKBACK_BUFFER_DAYS=600 (needed there because a daily-bar ROTATION
# signal can require up to a 252-trading-session/12-month moving-average
# lookback -- see that constant's own comment in that file), a
# swing-trading candidate's signal_fn realistically needs at most a few weeks
# of trailing context to compute typical short-hold technical indicators
# (e.g. a 20-day moving average, a 14-day RSI, a 10-day ATR) -- an order of
# magnitude less than a multi-month regime filter.
#
# 90 calendar days is the swing track's default: using the SAME
# typical trading-day fraction batch3 uses (~0.69, i.e. roughly 5
# trading sessions per 7 calendar days -- see that module's own
# HOLDOUT_LOOKBACK_BUFFER_DAYS comment), 90 calendar days yields ~62 real
# trading sessions of padding -- about 3x the longest indicator lookback
# named above (20 sessions), giving real margin without fetching months of
# extra MINUTE-bar data per candidate (a real cost distinct from batch3's
# daily-bar case: a minute-bar frame is roughly 390x larger per trading day
# than a daily-bar frame).
#
# This is a DEFAULT, not a hard requirement -- every caller of
# `reconcile_holdout_window` supplies its own already-fetched padded data, so
# this constant only matters as the SUGGESTED buffer size for whatever
# fetches that data (scripts/run_swing_mining.py); pass a candidate-specific
# override there if a specific signal's own trailing-context need is
# measured (not assumed) to exceed it -- mirroring how batch3's own 600-day
# figure is sized to cover its longest-lookback signals with margin (see
# that module's HOLDOUT_LOOKBACK_BUFFER_DAYS comment).
SWING_LOOKBACK_BUFFER_DAYS = 90

_INTRADAY_TRADE_COLUMNS = [
    "entry_datetime", "exit_datetime", "entry_px", "exit_px",
    "held_days", "reason", "gross_ret", "net_ret",
]  # matches intraday_backtest_engine.run_intraday_backtest's documented
   # trades_df contract exactly (that module's own docstring, "Returns
   # (trades_df, daily_returns)").


def _touched_day_reference_prices(
        bars: pd.DataFrame, entry_day, exit_day, exit_px: float) -> tuple[list, dict]:
    """Every unique calendar day present in `bars` within `[entry_day,
    exit_day]` (inclusive), and each one's reference price `ref(d)` --
    `exit_px` for `exit_day`, that day's own last-bar Close otherwise.
    Mirrors `intraday_backtest_engine.run_intraday_backtest`'s private
    `_record_legs` closure EXACTLY (not importable without modifying that
    module -- reproduced here since it is purely a function of `bars` and
    a trade's own entry/exit day boundaries; see `intraday_trade_stats`'s
    own docstring for this module's established "mirror the formula, cite
    the source" convention when direct reuse isn't possible)."""
    all_days = pd.unique(bars.index.date)  # bars is time-ascending -> order-preserving
    touched_days = [d for d in all_days if entry_day <= d <= exit_day]
    ref = {}
    for d in touched_days:
        if d == exit_day:
            ref[d] = exit_px
        else:
            day_bars = bars.loc[bars.index.date == d]
            ref[d] = float(day_bars.iloc[-1]["Close"])
    return touched_days, ref


def _per_day_gross_legs(
        entry_px: float, touched_days: list, ref: dict, direction: str) -> dict:
    """GROSS (pre cost/tax) per-day legs for one trade, mirroring
    `_record_legs`'s formula exactly: LONG uses the multiplicative ratio
    against the PREVIOUS day's own reference (chained day to day); SHORT
    uses the additive, FIXED-basis fraction against the original
    `entry_px` (never the previous day's mark, unlike long) -- see
    `intraday_backtest_engine.py`'s own docstring, "SHORT's trade-level
    convention," for the boundedness proof behind this asymmetry."""
    legs = {}
    prev_ref = entry_px
    for d in touched_days:
        curr_ref = ref[d]
        if direction == "long":
            legs[d] = curr_ref / prev_ref - 1.0
        else:
            legs[d] = (prev_ref - curr_ref) / entry_px
        prev_ref = curr_ref
    return legs


def _apply_net_ret_plug(
        legs: dict, touched_days: list, direction: str, net_ret: float) -> dict:
    """Overwrite the FINAL (exit-day) leg so the whole series reconstructs
    `net_ret` exactly -- mirrors `_record_legs`'s own cost/tax-realization
    step (LONG: solve via division/product; SHORT: solve via
    subtraction/sum). A single-day trade (`touched_days` has one element)
    reduces to `legs[only_day] = net_ret` directly (the interior
    product/sum over zero elements is the identity, 1.0/0.0), matching
    `intraday_backtest_engine.py`'s own documented same-day-round-trip
    case exactly."""
    legs = dict(legs)
    final_day = touched_days[-1]
    interior_days = touched_days[:-1]
    if direction == "long":
        interior_product = 1.0
        for d in interior_days:
            interior_product *= (1.0 + legs[d])
        legs[final_day] = (1.0 + net_ret) / interior_product - 1.0
    else:
        interior_sum = sum(legs[d] for d in interior_days)
        legs[final_day] = net_ret - interior_sum
    return legs


def reconcile_holdout_window(
    trades_df: pd.DataFrame,
    daily_returns: pd.Series,
    bars: pd.DataFrame,
    holdout_start: str,
    holdout_end: str,
    *,
    cost_bps: float,
    tax_haircut_rate: float = 0.0,
    direction: str = "long",
) -> tuple[pd.DataFrame, pd.Series]:
    """Slice a padded-window `(trades_df, daily_returns)` pair (as produced
    by one `intraday_backtest_engine.run_intraday_backtest` call over a
    lookback-padded span) down to the REAL holdout window
    `[holdout_start, holdout_end]` (both inclusive, "YYYY-MM-DD" strings).
    `bars` is the SAME padded OHLCV frame that was passed to
    `run_intraday_backtest` to produce `trades_df`/`daily_returns`.

    Returns `(real_trades_df, real_daily_returns)`:

    - `real_daily_returns`: `daily_returns` restricted to dates
      `>= holdout_start` and `<= holdout_end`. For a day NOT touched by a
      spanning trade, no re-pricing is needed -- every day's own mark is
      already honest on its own (see module docstring). **For a spanning
      trade's own touched days, this IS re-priced** (slicing dates alone
      would leave those values untouched and wrong): see the "spanning
      trade" section below for why and how.

    - `real_trades_df`: the subset of `trades_df` whose `exit_datetime`
      (converted to its own local calendar date) falls within
      `[holdout_start, holdout_end]`. A trade that both opened AND closed
      before `holdout_start` is pure warm-up and is dropped.

      **A SPANNING trade (entered before `holdout_start`, exits inside the
      real holdout window) is RE-PRICED, not kept as-is** -- keeping
      a spanning trade's ORIGINAL `net_ret`/`gross_ret` unchanged would
      be a serious bug: `net_ret` is a TRADE-LEVEL aggregate computed
      once, from the trade's ORIGINAL `entry_px` (a padding-period price)
      to `exit_px`. Unlike `daily_returns` (which already decomposes into
      one honest mark PER DAY -- see module docstring), `trades_df`'s
      `net_ret` has NO per-day decomposition; keeping it unchanged after
      slicing would let `intraday_trade_stats`/`holdout_net_edge_bps`
      silently include however much of the move happened during the
      PADDING period (potentially the entire move) as if it were real,
      out-of-sample holdout performance -- a genuine look-ahead-shaped
      leak into the very number the holdout-pass check is supposed to
      judge honestly. This is the SAME class of problem
      `run_batch3_mining._mark_to_market_spanning_trades` exists to solve
      for the daily rotation engine, and it needs essentially the SAME
      fix here.

      **`daily_returns` ALSO needs correcting for a spanning trade's own
      touched days, in addition to the trade-level fix above.** The claim that
      "`daily_returns` slicing needs no re-pricing" (module docstring)
      is true for a day untouched by any spanning trade, but NOT
      unconditionally true for the days a spanning trade itself touches:
      `intraday_backtest_engine`'s SHORT convention uses the trade's
      ORIGINAL (padding-period) `entry_px` as a FIXED denominator for
      EVERY leg, interior days included (see that module's own docstring,
      "SHORT's trade-level convention") -- those legs are still scaled
      against the wrong basis after a plain slice. LONG's interior legs
      are entry_px-independent (a rolling ratio against the PREVIOUS
      day's own mark) and so stay correct on their own, but BOTH
      directions share one more problem: the EXIT day's leg is a "plug"
      solved so the trade's full padding-through-holdout leg sequence
      reconstructs the ORIGINAL (not re-priced) `net_ret` -- reusing that
      plug value unchanged would silently feed pre-holdout P&L into
      `compute_shared_correction`'s cross-candidate correlation estimate
      via a bloated or shrunk exit-day mark, corrupting `dsr.
      num_independent_trials` for every OTHER candidate scored in the
      same batch, not just this one.

      This function therefore recomputes the spanning trade's own per-day legs TWICE,
      via `_touched_day_reference_prices`/`_per_day_gross_legs`/
      `_apply_net_ret_plug` (mirroring `_record_legs`'s exact formula --
      see those helpers' own docstrings): once from the trade's ORIGINAL
      entry (to know precisely what to divide back OUT of each already-
      combined `daily_returns` day -- `bars` gives every reference price
      needed), once from the SYNTHETIC holdout entry with the RE-PRICED
      `net_ret` (to know what to multiply back IN). This correctly
      handles a same-day co-trade too (a fresh position opening the exact
      day this one exits): dividing out the ORIGINAL leg from that day's
      COMBINED value isolates whatever the OTHER trade alone contributed,
      and multiplying the CORRECTED leg back in recombines it -- the same
      multiplicative same-day convention `intraday_backtest_engine.py`
      itself uses. A day exclusively held by this trade (provably true
      for every day strictly between its own entry and exit, since this
      engine holds at most one position at a time) has nothing else to
      divide out, so the "divide out, multiply in" operation reduces to a
      direct replacement there -- see `test_mining_intraday_holdout.py`'s
      genuine-marks tests for a spanning trade (long and short).

      A spanning trade is re-priced using the SAME convention
      `run_batch3_mining._mark_to_market_spanning_trades` already
      established and this pipeline already trusts (its own docstring:
      "re-priced as if freshly opened at the first REAL holdout session's
      open ... the same 'enter at Open' convention rotation_engine.py
      itself uses ... keeping the position's true real exit and exactly
      the ONE real round-trip cost paid at that exit"): `synthetic_entry_px`
      is the FIRST bar's Open within `[holdout_start, holdout_end]` in
      `bars`; `gross_ret`/`net_ret` are recomputed with
      `intraday_backtest_engine`'s own trade-level formula (`long_ret =
      exit_px/synthetic_entry_px - 1`, `gross_ret = long_ret if direction
      == "long" else -long_ret`, `net_ret = gross_ret - cost_bps/1e4`,
      then the same haircut-on-positive-net-only tax rule) -- no new cost
      is invented for the synthetic re-entry, matching batch3's own
      reasoning: no real transaction happens at a mark-to-market boundary,
      only the trade's one real exit did. `entry_px`/`entry_datetime` are
      updated to the synthetic values (for an honest audit trail); `held_
      days` is recomputed over the real trading days actually spanned
      within `[synthetic_entry_date, exit_date]`, using `bars`' own
      calendar. A trade FULLY inside the holdout window (entry date also
      `>= holdout_start`) is left completely unchanged -- no boundary to
      correct.

      A trade whose `exit_datetime` lands exactly on `holdout_start`'s
      date is treated as a spanning trade (re-priced), not dropped as pure
      warm-up -- a deliberate divergence from batch3's own boundary
      convention. batch3 drops that case because at ITS bar/price-level
      boundary, `synthetic_entry_date == exit_date` there could mean the
      synthetic entry price and the real exit price are the identical
      bar's switch price -- a zero-return leg with a real cost still
      charged for zero elapsed activity. That degenerate case cannot arise
      here even after re-pricing: `synthetic_entry_px` is `holdout_start`'s
      first bar's OPEN, while `exit_px` (via `intraday_backtest_engine
      .run_intraday_backtest`'s own `_close_trade`) is always some bar's
      CLOSE -- even when the exit happens to fall on that very first bar
      of `holdout_start`, Open and Close of the SAME bar are (in real
      market data) different prices reflecting genuine intraday movement,
      not an identical synthetic-entry/exit price pair.

    Raises ValueError if: `holdout_start` is after `holdout_end`; a
    spanning trade exists but `bars` has no rows within
    `[holdout_start, holdout_end]` (should never happen in practice: the
    caller always fetches `bars` THROUGH `holdout_end`, so an empty
    real-holdout slice of `bars` would mean the fetch itself was wrong);
    or a spanning trade's own `exit_datetime` is chronologically BEFORE the
    synthetic entry timestamp -- provably impossible when `bars` is
    genuinely the same frame that produced `trades_df` (see the inline
    comment at the check itself for the proof), but a defensive fail-loud
    guard against a caller passing a mismatched or truncated `bars`
    argument.
    """
    start = pd.Timestamp(holdout_start).date()
    end = pd.Timestamp(holdout_end).date()
    if start > end:
        raise ValueError(
            f"reconcile_holdout_window: holdout_start {start} is after "
            f"holdout_end {end}")

    if len(daily_returns) == 0:
        real_daily_returns = daily_returns
    else:
        daily_mask = (daily_returns.index >= start) & (daily_returns.index <= end)
        # .copy(): about to be mutated in place below for a spanning
        # trade's own touched days -- boolean-mask .loc already returns a
        # new object in practice, but an explicit copy makes that
        # non-aliasing guarantee unconditional rather than relying on
        # pandas-version behavior.
        real_daily_returns = daily_returns.loc[daily_mask].copy()

    if len(trades_df) == 0:
        return trades_df.copy(), real_daily_returns

    exit_dates = trades_df["exit_datetime"].dt.date
    entry_dates = trades_df["entry_datetime"].dt.date
    trade_mask = (exit_dates >= start) & (exit_dates <= end)
    real_trades_df = trades_df.loc[trade_mask.to_numpy()].copy()

    spanning_mask = (entry_dates < start) & trade_mask
    spanning_idx = trades_df.index[spanning_mask.to_numpy()]
    if len(spanning_idx) > 0:
        holdout_bars = bars.loc[
            (bars.index.date >= start) & (bars.index.date <= end)]
        if holdout_bars.empty:
            raise ValueError(
                "reconcile_holdout_window: a spanning trade exists but "
                f"`bars` has no rows within [{start}, {end}] -- bars must "
                "be fetched at least through holdout_end")
        synthetic_entry_ts = holdout_bars.index[0]
        synthetic_entry_px = float(holdout_bars.iloc[0]["Open"])
        holdout_days = pd.unique(holdout_bars.index.date)

        for idx in spanning_idx:
            exit_dt = trades_df.at[idx, "exit_datetime"]
            # Defensive invariant: a spanning trade's real exit must be AT
            # OR AFTER the synthetic entry (a mismatched/truncated `bars`
            # frame -- one that does not actually cover the trade's own
            # exit_datetime -- could put the synthetic entry chronologically
            # AFTER the real exit, producing a temporally impossible,
            # silently-wrong return). Under this function's own documented
            # contract (`bars` is THE SAME frame `run_intraday_backtest`
            # used to produce `trades_df`), this can be proven never to
            # happen: `bars` is time-ascending, `exit_dt` is itself a row of
            # `bars` with date >= start (by `trade_mask`'s own definition),
            # and `holdout_bars` (a date-filtered, order-preserving subset
            # of the SAME `bars`) therefore contains `exit_dt`'s row at or
            # after its own first row (test_reconcile_never_produces_
            # synthetic_entry_after_real_exit sweeps every entry/exit bar
            # pair across several holdout_start offsets, both directions).
            # But a caller passing a
            # `bars` frame that does NOT actually match `trades_df` (e.g.
            # truncated, or from a different fetch) breaks that proof's
            # premise -- fail loudly here rather than silently produce a
            # nonsensical negative-duration return in that case.
            if exit_dt < synthetic_entry_ts:
                raise ValueError(
                    f"reconcile_holdout_window: trade at index {idx} exits "
                    f"({exit_dt}) before the synthetic holdout entry "
                    f"({synthetic_entry_ts}) -- `bars` must be the SAME "
                    f"frame run_intraday_backtest used to produce "
                    f"trades_df/daily_returns; a mismatched or truncated "
                    f"`bars` argument can violate this invariant")

            exit_px = float(trades_df.at[idx, "exit_px"])
            long_ret = exit_px / synthetic_entry_px - 1.0
            gross_ret = long_ret if direction == "long" else -long_ret
            net_ret = gross_ret - cost_bps / 1e4
            if net_ret > 0.0 and tax_haircut_rate:
                net_ret *= (1.0 - tax_haircut_rate)

            exit_date = exit_dt.date()
            held_days = int(np.sum(
                (holdout_days >= synthetic_entry_ts.date())
                & (holdout_days <= exit_date)))

            # --- repair daily_returns for this trade's own touched days --
            # see this function's own docstring, "daily_returns ALSO needs
            # correcting," for the full reasoning. Recompute the ORIGINAL
            # per-day legs (to
            # know what to divide back OUT of each already-combined day)
            # and the CORRECTED per-day legs (to know what to multiply
            # back IN), then swap them in real_daily_returns for every day
            # this trade actually touches within the real holdout window.
            orig_entry_px = float(trades_df.at[idx, "entry_px"])
            orig_entry_day = trades_df.at[idx, "entry_datetime"].date()
            orig_days, orig_ref = _touched_day_reference_prices(
                bars, orig_entry_day, exit_date, exit_px)
            orig_legs = _apply_net_ret_plug(
                _per_day_gross_legs(orig_entry_px, orig_days, orig_ref, direction),
                orig_days, direction, float(trades_df.at[idx, "net_ret"]))

            new_days, new_ref = _touched_day_reference_prices(
                bars, synthetic_entry_ts.date(), exit_date, exit_px)
            new_legs = _apply_net_ret_plug(
                _per_day_gross_legs(synthetic_entry_px, new_days, new_ref, direction),
                new_days, direction, net_ret)

            for d in new_days:
                combined = real_daily_returns.loc[d]
                without_original = (1.0 + combined) / (1.0 + orig_legs[d]) - 1.0
                real_daily_returns.loc[d] = (
                    (1.0 + without_original) * (1.0 + new_legs[d]) - 1.0)

            real_trades_df.at[idx, "entry_datetime"] = synthetic_entry_ts
            real_trades_df.at[idx, "entry_px"] = synthetic_entry_px
            real_trades_df.at[idx, "gross_ret"] = gross_ret
            real_trades_df.at[idx, "net_ret"] = net_ret
            real_trades_df.at[idx, "held_days"] = held_days

    real_trades_df = real_trades_df.reset_index(drop=True)
    return real_trades_df, real_daily_returns


def pool_intraday_trades(trades_by_ticker: dict) -> pd.DataFrame:
    """Concatenate every ticker's trades_df (`run_intraday_backtest` output
    shape) into one pooled DataFrame tagged with a `ticker` column, sorted
    chronologically by `entry_datetime` (ticker name as a deterministic
    tie-break) -- the same design as `mining.pooling.pool_trades`, adapted
    for this engine's own column names (see module docstring for why
    `pool_trades` itself cannot be reused unmodified: it hardcodes
    `entry_date`, this engine's trades_df has `entry_datetime`).

    A ticker that produced zero trades contributes nothing (not an error --
    "this ticker never triggered the signal" is a legitimate real outcome,
    same convention as `pool_trades`). Raises ValueError only if
    `trades_by_ticker` itself is empty (a caller passing no tickers at all
    is a bug, distinct from tickers that ran and produced zero trades).
    """
    if not trades_by_ticker:
        raise ValueError(
            "pool_intraday_trades: trades_by_ticker is empty -- no tickers "
            "were passed in at all")

    frames = []
    for ticker, trades_df in trades_by_ticker.items():
        if len(trades_df) == 0:
            continue
        tagged = trades_df.copy()
        tagged["ticker"] = ticker
        frames.append(tagged)

    if not frames:
        sample = next(iter(trades_by_ticker.values()))
        empty = pd.DataFrame({
            col: pd.Series(dtype=sample.dtypes.get(col, "object"))
            for col in _INTRADAY_TRADE_COLUMNS
        })
        empty["ticker"] = pd.Series(dtype="object")
        return empty

    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.sort_values(
        ["entry_datetime", "ticker"], kind="stable").reset_index(drop=True)
    return pooled


def pool_swing_daily_returns(daily_returns_by_ticker: dict) -> pd.Series:
    """Sum (not average) each ticker's own `daily_returns` Series
    (`run_intraday_backtest` output shape, indexed by `datetime.date`) onto
    the union of every ticker's touched dates. Mirrors
    `mining.pooling.pooled_daily_returns_by_exit_day`'s own SUM convention
    exactly -- see that module's docstring for why summing (not averaging)
    is the right aggregation for a multi-ticker family's combined daily P&L
    ("this family's aggregate realized P&L, across every ticker it touched,
    on each calendar day"). Unlike that function, this one does not need a
    separate OHLC index per ticker to derive the series from scratch --
    `run_intraday_backtest` already produces one directly -- so this is a
    plain union+sum, not a reimplementation of trade-to-daily derivation.

    Raises ValueError if `daily_returns_by_ticker` is empty.
    """
    if not daily_returns_by_ticker:
        raise ValueError(
            "pool_swing_daily_returns: daily_returns_by_ticker is empty -- "
            "no tickers were passed in at all")

    union_index = None
    for series in daily_returns_by_ticker.values():
        union_index = (series.index if union_index is None
                        else union_index.union(series.index))
    union_index = union_index.sort_values()

    combined = pd.Series(0.0, index=union_index)
    for series in daily_returns_by_ticker.values():
        combined = combined.add(
            series.reindex(union_index, fill_value=0.0), fill_value=0.0)
    return combined


def intraday_trade_stats(trades_df: pd.DataFrame) -> dict:
    """`{trade_count, net_edge_bps, sharpe, skew, kurtosis}` computed from an
    intraday `trades_df` (raw single-ticker or pooled via
    `pool_intraday_trades`), using the EXACT SAME formula as
    `scripts/run_batch3_mining.py`'s private `_trade_stats`:
    `net_edge_bps = mean(net_ret) * 1e4`; `sharpe = mean/std` (per-trade,
    ddof=1, 0.0 below 2 trades or zero/non-finite std); `skew`/`kurtosis` via
    `scipy.stats.skew`/`kurtosis(..., bias=True)` (the plug-in/moment
    convention `dsr.py` requires -- see that module's own warning about
    mixing bias conventions), floored at `kurtosis = max(kurtosis, 1.0 +
    skew**2)` -- the same Pearson-identity floor `_trade_stats` applies, and
    which `mining.pooling.pooled_trade_stats` (the OTHER existing formula in
    this pipeline) omits. See module docstring for why this floor, not
    `pooled_trade_stats`'s formula, is the one mirrored here: omitting it
    lets a real pooled sample raise an unhandled `ValueError` deep inside
    `dsr.deflated_sharpe_ratio` instead of producing a valid (if
    conservative) score.

    `_trade_stats` is private to the `run_batch3_mining.py` entry-point
    script, and this library module does not import from `scripts/` -- but
    the formula itself is column-name-agnostic (only ever
    touches `trades["net_ret"]`), so reproducing it verbatim here carries no
    behavioral drift risk.
    """
    count = int(len(trades_df))
    if count == 0:
        return {"trade_count": 0, "net_edge_bps": 0.0, "sharpe": 0.0,
                "skew": 0.0, "kurtosis": 3.0}

    returns = pd.to_numeric(trades_df["net_ret"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(returns).all():
        raise ValueError("intraday_trade_stats: trade returns must be finite")

    edge = float(np.mean(returns) * 1e4)
    std = float(np.std(returns, ddof=1)) if count >= 2 else 0.0
    if count < 2 or std <= 0 or not math.isfinite(std):
        return {"trade_count": count, "net_edge_bps": edge, "sharpe": 0.0,
                "skew": 0.0, "kurtosis": 3.0}

    skew = float(scipy.stats.skew(returns, bias=True))
    kurtosis = float(scipy.stats.kurtosis(returns, fisher=False, bias=True))
    if not math.isfinite(skew) or not math.isfinite(kurtosis):
        skew, kurtosis = 0.0, 3.0
    kurtosis = max(kurtosis, 1.0 + skew ** 2)

    return {"trade_count": count, "net_edge_bps": edge,
            "sharpe": float(np.mean(returns) / std), "skew": skew,
            "kurtosis": kurtosis}
