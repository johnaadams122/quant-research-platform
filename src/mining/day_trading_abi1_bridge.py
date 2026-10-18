"""Day-trading (intraday/minute-bar) -> ABI-1 account-curve bridge.

What this module does: `scripts/
run_abi1_mining.py` (the ABI-1 scoring entry point) discovers strategy modules via
`scripts/run_batch5_mining.py`'s `discover_modules`/`validate_module_contract`,
whose default OHLC loader (`run_batch5_mining.default_ohlc_loader` ->
`data_pipeline.fetch_ohlc`) only ever fetches DAILY bars -- there is no
intraday-timeframe path anywhere in that generic discovery/loading
mechanism. This module converts a day-trading candidate's real
`intraday_backtest_engine.run_intraday_backtest` output into the SAME
`{"strategy_daily", "in_market", "trades_df"}` account-curve dict shape
`mining.abi1_driver.evaluate_family` accepts from any caller, not only from
its own three named builders (`build_single_asset_curve`/`build_rotation_curve`/
`build_pooled_curve`).

A small, dedicated, REUSABLE module (not embedded in one family's bespoke
driver script) because the same need -- an intraday/minute-bar signal with
no daily-timeframe path into ABI-1 -- applies to every intraday-timeframe
family, not just one; a bespoke driver for any of them can import
`build_day_trading_curve` directly rather than re-deriving
this aggregation from scratch.

Scope: the DAY-TRADING track specifically (`max_hold_days=1` -- every
trade opens AND closes within the SAME session, this track's own mechanical
invariant, enforced independently by `run_day_trading_mining.
_validate_day_trading_candidate`). `build_day_trading_curve` asserts this
invariant defensively (raises on any trade whose entry/exit calendar day
differ) rather than silently misfiling a violation -- a genuinely
multi-day-hold intraday/swing family would need a different aggregation
(spreading a trade's return across every day it was held, the same
mark-to-market decomposition `intraday_backtest_engine.py`'s own docstring
derives), not this one.

Never imports or modifies `intraday_backtest_engine.py`, `abi1_driver.py`,
or any `day_trading_*` module -- additive only, consumes their outputs.
"""
from __future__ import annotations

import pandas as pd

from mining.pooling import pool_trades


def build_day_trading_curve(trades_df: pd.DataFrame,
                             calendar_index: pd.DatetimeIndex) -> dict:
    """Aggregate a day-trading track's real per-trade `trades_df` (the
    first return value of `intraday_backtest_engine.run_intraday_backtest`)
    into `{"strategy_daily", "in_market", "trades_df"}`, aligned to
    `calendar_index` -- the real trading-day calendar the caller's own
    intraday bars actually span (e.g. built from `bars.index.date`,
    deduplicated and sorted).

    `calendar_index` MUST already be tz-naive and midnight-normalized (the
    SAME convention `run_batch3_mining._normalize_ohlc` gives every real
    daily OHLC benchmark frame, e.g. SPY) -- `strategy_daily`/`in_market`
    are built directly on it, so a caller's own benchmark-alignment join
    (`abi1_driver.evaluate_family` -> `account_relative.evaluate` ->
    `log_excess`) lines sessions up correctly rather than silently
    producing an empty intersection because one side carries a timezone
    and the other does not.

    Per calendar day `d`:
      - `strategy_daily[d]` = every trade whose `exit_datetime` (and, by
        this function's own enforced invariant below, `entry_datetime`
        too) falls on day `d`, COMBINED MULTIPLICATIVELY in chronological
        order (`product(1 + net_ret) - 1`) -- a real day-trading session
        can hold, exit, and re-enter more than once (this track's
        single-position-at-a-time invariant forbids two OPEN positions at
        once, not two ROUND TRIPS in one session), so a calendar day can
        carry more than one closed trade. This is the SAME same-day
        multi-trade combination rule `intraday_backtest_engine.py`'s own
        module docstring documents for its internal `per_day_legs`
        aggregation -- reused here at the trades_df level rather than
        re-derived differently. 0.0 on a day with no closed trade.
      - `in_market[d]` = True iff at least one trade's `entry_datetime`
        (equivalently `exit_datetime`, enforced equal below) falls on day
        `d`. False elsewhere.
      - `trades_df` is a COPY of the input with two new columns added,
        `entry_date`/`exit_date` (tz-naive, midnight-normalized -- the
        SAME calendar-day representation `calendar_index` itself uses),
        derived from `entry_datetime`/`exit_datetime`'s own calendar day.
        These two columns are NOT optional cosmetics:
        `mining.account_relative.netted_tax_factors` (reached through
        `abi1_driver.evaluate_family` -> `account_relative.evaluate`)
        requires them BY NAME to determine each trade's long-term/
        short-term holding-period character, and raises ValueError without
        them (exercised end to end through `evaluate_family` in this module's
        own test file). `backtest_engine.run_backtest`'s own (daily) trades_df
        already carries them under exactly these names for the identical
        reason; this bridge adds the equivalent for the intraday engine's
        differently-shaped trades_df (`entry_datetime`/`exit_datetime`,
        minute-precision) rather than silently omitting them. The original
        `trades_df` object passed in is never mutated.

    Raises ValueError if any trade's `entry_datetime`/`exit_datetime`
    calendar days differ (a real violation of this track's own
    `max_hold_days=1` mechanical invariant -- see module docstring; fail
    loud rather than silently file the trade under whichever day happens
    to match) or if a trade's day is not present in `calendar_index` (bars
    and trades_df out of sync -- a real caller bug, not a silently-dropped
    contribution).
    """
    daily = pd.Series(0.0, index=calendar_index, name="strategy_daily")
    in_market = pd.Series(False, index=calendar_index, name="in_market")

    entry_dates = []
    exit_dates = []
    for _, trade in trades_df.iterrows():
        entry_day = pd.Timestamp(pd.Timestamp(trade["entry_datetime"]).date())
        exit_day = pd.Timestamp(pd.Timestamp(trade["exit_datetime"]).date())
        if entry_day != exit_day:
            raise ValueError(
                "build_day_trading_curve: trade entry/exit calendar day "
                f"mismatch ({entry_day.date()} -> {exit_day.date()}) -- this "
                "aggregation assumes every day-trading trade closes the SAME "
                "session it opened (max_hold_days=1); a mismatch means the "
                "signal_fn/engine call that produced this trades_df violated "
                "that mechanical invariant")
        if entry_day not in daily.index:
            raise ValueError(
                f"build_day_trading_curve: trade day {entry_day.date()} is "
                "not in the supplied calendar_index -- bars and trades_df "
                "are out of sync")
        net_ret = float(trade["net_ret"])
        daily.loc[entry_day] = (1.0 + daily.loc[entry_day]) * (1.0 + net_ret) - 1.0
        in_market.loc[entry_day] = True
        entry_dates.append(entry_day)
        exit_dates.append(exit_day)

    trades_df = trades_df.copy()
    trades_df["entry_date"] = pd.Series(
        entry_dates, index=trades_df.index, dtype="datetime64[ns]")
    trades_df["exit_date"] = pd.Series(
        exit_dates, index=trades_df.index, dtype="datetime64[ns]")

    return {
        "strategy_daily": daily,
        "in_market": in_market,
        "trades_df": trades_df,
    }


def _union_aligned(series_by_key: dict, *, how: str) -> pd.Series:
    """Same union-calendar/reindex-zero-fill/combine helper
    `scripts/run_batch5_mining.py`'s own private `_aligned` implements --
    reimplemented here (not imported) because that function lives in a
    `scripts/` entry-point module, and this bridge is a `src/mining/`
    library module every bespoke driver script imports; a library module
    importing FROM a script would invert this repo's own layering. Same
    two behaviors: `how="sum"` sums every member's value (0.0 where a
    member has no observation that day); `how="mean"` additionally divides
    by the member count when there is more than one."""
    union = pd.DatetimeIndex([])
    for series in series_by_key.values():
        union = union.union(pd.DatetimeIndex(series.index))
    union = union.sort_values()
    total = pd.Series(0.0, index=union)
    for series in series_by_key.values():
        total = total.add(series.reindex(union, fill_value=0.0), fill_value=0.0)
    if how == "mean" and len(series_by_key) > 1:
        total = total / float(len(series_by_key))
    return total


def build_pooled_day_trading_curve(curves_by_ticker: dict) -> dict:
    """Combine several tickers' own `build_day_trading_curve` outputs into
    ONE pooled account curve for a multi-ticker day-trading family (e.g.
    `fvg_reversal_cascade_continuation_daytrade`, pooled across QQQ/SPY/
    IWM) -- the day-trading-track equivalent of `mining.abi1_driver.
    build_pooled_curve`, reusing that function's own combination
    rule rather than inventing a new one for this track:

    - `in_market` = OR across every member's own `in_market` mask, each
      reindexed onto the pool's UNION calendar with `fill_value=False`
      before the OR (implemented as "sum of 0.0/1.0 across members > 0",
      the identical trick `build_pooled_curve` itself uses so `in_market`
      and `strategy_daily` are guaranteed to share one union calendar by
      construction).
    - `strategy_daily` = equal-weight MEAN of every member's own
      `strategy_daily` sleeve on that same union calendar, `0.0`-filled
      for a member absent (or simply not traded) that session -- never a
      naive sum, which would overstate the account's real capital
      exposure on a day more than one member is simultaneously held.
    - `trades_df` = `mining.pooling.pool_trades({ticker: that member's own
      trades_df})` -- every member's OWN undiluted trades, concatenated
      and ticker-tagged. Reusable directly (not `mining.intraday_holdout.
      pool_intraday_trades`, that module's own day-trading-shaped
      equivalent) because `build_day_trading_curve` already adds
      `entry_date`/`exit_date` columns to its own `trades_df` output,
      making it the SAME shape `pool_trades` already expects from
      `backtest_engine.run_backtest`.

    `curves_by_ticker` maps each pool member's ticker to its OWN
    `build_day_trading_curve` return value (built from that ticker's own
    real intraday backtest -- never a shared signal_fn's single run reused
    across tickers). Raises ValueError if `curves_by_ticker` is empty (a
    caller passing no pool members at all is a bug, distinct from a member
    that ran and produced zero trades, which contributes 0.0/False/nothing
    and is not an error -- same convention as `pool_trades` and
    `build_pooled_curve` themselves).
    """
    if not curves_by_ticker:
        raise ValueError(
            "build_pooled_day_trading_curve: curves_by_ticker is empty -- "
            "no pool members were passed in at all")

    in_market = _union_aligned(
        {ticker: curve["in_market"].astype(float)
         for ticker, curve in curves_by_ticker.items()},
        how="sum") > 0
    in_market.name = "in_market"

    strategy_daily = _union_aligned(
        {ticker: curve["strategy_daily"] for ticker, curve in curves_by_ticker.items()},
        how="mean")
    strategy_daily.name = "strategy_daily"

    trades_df = pool_trades(
        {ticker: curve["trades_df"] for ticker, curve in curves_by_ticker.items()})

    return {
        "strategy_daily": strategy_daily,
        "in_market": in_market,
        "trades_df": trades_df,
    }
