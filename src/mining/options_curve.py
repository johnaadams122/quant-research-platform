"""Translation layer from an options cycle driver's result
({nav_daily, in_market, realizations} -- wheel_engine.run_wheel/run_pmcc/
run_put_ratio_spread) into the ABI-1 shape-agnostic scoring contract
({strategy_daily, in_market, trades_df}) every abi1_driver.py curve
builder already produces (options-chain simulator).

account_relative.py consumes this shape as-is: evaluate/score_account/
risk_refusals/holdout_gate all take abstract strategy_daily/in_market/
trades_df with no knowledge of what produced them -- the same property
abi1_driver.py's build_rotation_curve/build_pooled_curve rely on for equity
shapes (an extra "asset" trades_df column and a diluted pooled
strategy_daily both pass through untouched).
"""
from __future__ import annotations

import pandas as pd

from . import pooling
from .abi1_driver import _aligned

_TRADES_COLUMNS = ["entry_date", "exit_date", "entry_px", "exit_px", "held",
                   "reason", "gross_ret", "net_ret"]


def _empty_trades_df() -> pd.DataFrame:
    return pd.DataFrame({
        "entry_date": pd.Series(dtype="datetime64[ns]"),
        "exit_date": pd.Series(dtype="datetime64[ns]"),
        "entry_px": pd.Series(dtype="float64"),
        "exit_px": pd.Series(dtype="float64"),
        "held": pd.Series(dtype="int64"),
        "reason": pd.Series(dtype="object"),
        "gross_ret": pd.Series(dtype="float64"),
        "net_ret": pd.Series(dtype="float64"),
    })


def build_options_curve(cycle_result: dict) -> dict:
    """`cycle_result` is the dict returned by run_wheel/run_pmcc/
    run_put_ratio_spread: {"nav_daily": pd.Series, "in_market": pd.Series,
    "realizations": list[dict]}.

    `strategy_daily = nav_daily.pct_change()`, with the first in-market
    day set to 0.0 (there is no prior day to diff against; that day's own
    move is already priced into the cycle's own day-0 NAV construction,
    matching mark_to_market_daily_returns' own entry-bar treatment
    elsewhere in this codebase, not a separate realized return to book
    here).

    `trades_df` is built by chopping `realizations` into rows:
    `entry_px`/`exit_px` are each realization's own `nav_entry`/`nav_exit`
    (dimensionless per-cycle NAV-ratio units -- safe, since
    account_relative.netted_tax_factors only ever uses
    `net_ret = exit_px/entry_px - 1` as a RATIO). `gross_ret` and `net_ret`
    are EQUAL for every options
    trade (cost is already asymmetrically baked into the NAV path itself
    -- there is no separate cost-free NAV series to diff against). `held`
    is CALENDAR days between entry/exit (`reason` is free text -- never
    branched on downstream, same as build_rotation_curve's own extra
    "asset" column).

    Returns the standard `{strategy_daily, in_market, trades_df}` dict.
    `in_market` is passed through from `cycle_result` UNCHANGED (already
    built on the identical index as `nav_daily` by every driver).
    """
    nav_daily = cycle_result["nav_daily"]
    in_market = cycle_result["in_market"]
    realizations = cycle_result["realizations"]

    if len(nav_daily) == 0:
        return {"strategy_daily": pd.Series(dtype=float), "in_market": in_market,
                "trades_df": _empty_trades_df()}

    strategy_daily = nav_daily.pct_change()
    strategy_daily.iloc[0] = 0.0

    rows = []
    for rl in realizations:
        entry_px, exit_px = rl["nav_entry"], rl["nav_exit"]
        gross_ret = exit_px / entry_px - 1.0
        rows.append({
            "entry_date": rl["entry_date"], "exit_date": rl["exit_date"],
            "entry_px": entry_px, "exit_px": exit_px,
            "held": (rl["exit_date"] - rl["entry_date"]).days,
            "reason": rl["reason"], "gross_ret": gross_ret, "net_ret": gross_ret,
        })
    trades_df = (pd.DataFrame(rows, columns=_TRADES_COLUMNS) if rows
                else _empty_trades_df())

    return {"strategy_daily": strategy_daily, "in_market": in_market,
            "trades_df": trades_df}


def pool_options_curves(per_ticker_curve: dict) -> dict:
    """Aggregates ALREADY-BUILT per-member options curves
    ({strategy_daily, in_market, trades_df}, from build_options_curve)
    into one pooled family curve -- the SAME aggregation
    abi1_driver.build_pooled_curve uses (OR-of-in_market reindexed onto
    the union calendar with fill_value=False; equal-weight MEAN of
    strategy_daily on that same calendar via the shared _aligned helper;
    trades_df via pooling.pool_trades), adapted to consume pre-built
    curves rather than build them from a SignalFn+frame pair --
    abi1_driver.build_pooled_curve cannot be reused verbatim here, since
    the Wheel/PMCC/ratio-spread drivers are calendar/DTE-rule-driven, not
    backtest_engine.SignalFn-based.

    Aggregation semantics: each member's in_market contributes 0.0/False when
    absent that session, never a spurious True; strategy_daily is
    0.0-filled for an absent member before averaging.
    """
    members = list(per_ticker_curve.keys())
    if not members:
        raise ValueError(
            "pool_options_curves: per_ticker_curve is empty -- no pool "
            "members were passed in at all")

    in_market = _aligned(
        {symbol: curve["in_market"].astype(float)
         for symbol, curve in per_ticker_curve.items()}, how="sum") > 0
    in_market.name = "in_market"

    strategy_daily = _aligned(
        {symbol: curve["strategy_daily"] for symbol, curve in per_ticker_curve.items()},
        how="mean")

    trades_df = pooling.pool_trades(
        {symbol: curve["trades_df"] for symbol, curve in per_ticker_curve.items()})

    return {"strategy_daily": strategy_daily, "in_market": in_market,
            "trades_df": trades_df}
