"""ABI-1 account-curve builders.

Part of the ABI-1 full-history account-relative scoring driver. Builds
`(strategy_daily, in_market, trades_df)` for every shape family in the
batch5 catalog:

* `SINGLE_SHAPES` (`single_long`, `single_inverse_product`,
  `panel_breadth`) -- exactly ONE traded instrument -> `build_single_asset_curve`.
* `rotation_two_asset` -- a shared-calendar two-asset rotation ->
  `build_rotation_curve`.
* `POOLED_SHAPES` (`pooled_single_asset`, `event_pooled`) -- many traded
  instruments under one hypothesis -> `build_pooled_curve` (see that
  function's own docstring for the full argument).

Every builder in this module returns the SAME dict shape:
`{"strategy_daily": pd.Series, "in_market": pd.Series, "trades_df":
pd.DataFrame}` -- the contract `evaluate_family` dispatches on,
shape-agnostically, across every builder this module contains.

`build_single_asset_curve`/`build_rotation_curve` run their underlying
engine with `tax_haircut_rate=0.0`. ABI-1's own tax model
(`account_relative.evaluate(tax_policy="realization", ...)` ->
`netted_tax_factors`) applies tax itself, working from the PRE-tax
`trades_df["net_ret"]` -- passing a nonzero `tax_haircut_rate` into the
engine here would double-tax every winning trade. `build_pooled_curve`'s
own caller must instead score under `tax_policy="none"` -- see that
function's docstring for why.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Optional

import pandas as pd

from . import (account_relative, benchmark_relative, constants,
               cost_tax_inputs, evidence, trial_ledger)
from .backtest_engine import SignalFn, run_backtest
from .benchmark_relative import mark_to_market_daily_returns
from .pooling import pool_trades
from .rotation_engine import TargetFn, run_two_asset_rotation


def _aligned(series_by_key: Mapping[str, pd.Series], *, how: str) -> pd.Series:
    """Union-calendar reindex/zero-fill/combine, copied verbatim from
    `scripts/run_batch5_mining.py`'s own `_aligned` rather than imported across the script/package
    boundary: `run_batch5_mining.py` is a top-level SCRIPT that can also run
    as `__main__`, so a `sys.path` import of it from a PACKAGE module (this
    file, imported by production code) risked the module being
    loaded twice under two different `sys.modules` keys
    (`"__main__"` and `"run_batch5_mining"`) if any future process both runs
    it as an entry point and imports `mining.abi1_driver` -- fully
    re-executing it and producing two non-identical copies of everything it
    defines. This ~10-line helper is small and stable enough to duplicate
    instead; if `run_batch5_mining._aligned`'s own definition ever changes,
    this copy needs a matching update (grep both names together first)."""
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


def build_single_asset_curve(full_frame: pd.DataFrame, signal_fn: SignalFn, *,
                              cost_bps: float) -> dict:
    """Account curve for the `SINGLE_SHAPES` family -- exactly one traded
    instrument, run over the FULL frame (not a windowed bucket).

    Runs `signal_fn` through `backtest_engine.run_backtest` (with
    `tax_haircut_rate=0.0`, see module docstring) to get `(trades_df,
    in_market)` directly, then feeds the same frame and trade list to
    `benchmark_relative.mark_to_market_daily_returns` (same `cost_bps`) to
    get `strategy_daily`.

    Returns `{"strategy_daily": pd.Series, "in_market": pd.Series,
    "trades_df": pd.DataFrame}`, all three aligned to `full_frame.index`
    (`trades_df` is `run_backtest`'s own per-trade frame).
    """
    trades_df, in_market = run_backtest(
        full_frame, signal_fn, cost_bps=cost_bps, tax_haircut_rate=0.0)
    strategy_daily = mark_to_market_daily_returns(
        full_frame, trades_df, cost_bps=cost_bps)
    return {
        "strategy_daily": strategy_daily,
        "in_market": in_market,
        "trades_df": trades_df,
    }


def _align_rotation_frames(asset_frames: Mapping[str, pd.DataFrame],
                            signal_frames: Optional[Mapping[str, pd.DataFrame]]
                            ) -> tuple:
    """Full-history equivalent of `run_batch3_mining._normalize_rotation_
    inception` + `_rotation_bucket`'s own alignment: a real batch5 rotation family can have
    TWO different kinds of calendar mismatch, and `rotation_engine.run_two_asset_rotation`
    itself refuses BOTH unconditionally (it requires every frame it is
    handed -- traded AND signal-only -- to share one identical calendar).

    (1) The two TRADED legs can list on different dates (a real
    recently-listed-instrument shape, the same one pooled families already
    guard against via MIN_POOL_COVERAGE). Trimmed to their COMMON
    INCEPTION (`max` of the two starts) here; an INTERIOR mismatch that
    survives that trim is a real data defect between two same-exchange
    instruments and is NOT silently patched by intersecting the legs
    against each other -- it fails closed, matching
    `_rotation_bucket`'s own "traded legs must share an identical
    calendar" precedent.

    (2) A SIGNAL-only frame (e.g. `month_end_rebalancing_flow_rotation`'s
    IEF) is allowed a DIFFERENT exact calendar from the traded legs --
    only its OVERLAP matters, so it is intersected onto the (already
    matched) traded calendar rather than required to match it exactly,
    again mirroring `_rotation_bucket`'s own signal-frame handling.

    Returns `(aligned_assets, aligned_signals)`, both dicts of frames
    sliced onto the SAME final calendar -- `aligned_signals` is `{}` when
    `signal_frames` is falsy."""
    common_inception = max(frame.index.min() for frame in asset_frames.values())
    trimmed_assets = {
        symbol: frame.loc[frame.index >= common_inception].copy()
        for symbol, frame in asset_frames.items()
    }
    calendars = list(trimmed_assets.values())
    asset_calendar = calendars[0].index
    for frame in calendars[1:]:
        if not frame.index.equals(asset_calendar):
            raise ValueError(
                "build_rotation_curve: the two traded legs must share an "
                f"identical calendar after trimming to their common "
                f"inception ({common_inception.date()}) -- a real mismatch "
                "between two same-exchange instruments is a data defect "
                "worth surfacing, not silently patched by intersecting "
                "the two legs against each other")
    if not signal_frames:
        return trimmed_assets, {}
    common = asset_calendar
    for frame in signal_frames.values():
        common = common.intersection(frame.index)
    if len(common) < 2:
        raise ValueError(
            "build_rotation_curve: the signal-only frame(s) share fewer "
            "than 2 sessions with the traded legs' aligned calendar")
    aligned_assets = {symbol: frame.loc[common].copy()
                      for symbol, frame in trimmed_assets.items()}
    aligned_signals = {symbol: frame.loc[common].copy()
                       for symbol, frame in signal_frames.items()}
    return aligned_assets, aligned_signals


def build_rotation_curve(asset_frames: Mapping[str, pd.DataFrame],
                          target_fn: TargetFn, *,
                          cost_bps_by_asset: Mapping[str, float],
                          signal_frames: Optional[Mapping[str, pd.DataFrame]] = None
                          ) -> dict:
    """Account curve for the `rotation_two_asset` shape.

    `signal_frames`: a real batch5 family (`month_end_rebalancing_flow_rotation`)
    declares a signal-only ticker (IEF) whose `make_target_fn` closure
    reads it straight out of the engine's own `truncated` dict
    (`truncated["IEF"]`) -- with no `signal_frames` wired through, that
    read would KeyError the moment a real driver ran this family. Both
    `asset_frames` and `signal_frames` are aligned to one common calendar
    first via `_align_rotation_frames` (see its own docstring for the two
    kinds of mismatch it handles) before anything is handed to the engine,
    so `run_two_asset_rotation`'s own unconditional identical-calendar
    check never fires on a caller-supplied full-history frame.

    Runs `rotation_engine.run_two_asset_rotation` (with
    `tax_haircut_rate=0.0`, see module docstring) to get `(trades_df,
    active_asset)`. `active_asset` is `None` on a session the rotation
    holds cash and an asset key otherwise. The account mask also includes
    exit sessions: an exit to cash still owns the overnight price move.

    Each leg is marked through its actual recorded exit fill. On a switch,
    the old leg's overnight factor and the new leg's intraday factor are
    sequential investments of the same capital, so their factors multiply.
    Cost follows the once-at-exit daily cost convention.

    Returns `{"strategy_daily": pd.Series, "in_market": pd.Series,
    "trades_df": pd.DataFrame}`. Unlike `build_single_asset_curve`'s
    `trades_df`, this `trades_df` is the rotation engine's own segments
    frame and carries an extra `"asset"` column.
    """
    aligned_assets, aligned_signals = _align_rotation_frames(
        asset_frames, signal_frames)
    trades_df, active_asset = run_two_asset_rotation(
        aligned_assets, target_fn, cost_bps_by_asset=cost_bps_by_asset,
        tax_haircut_rate=0.0, signal_frames=aligned_signals or None)
    in_market = active_asset.notna()
    if not trades_df.empty:
        in_market.loc[pd.DatetimeIndex(trades_df["exit_date"])] = True
    in_market.name = "in_market"

    strategy_daily = None
    for asset_key, frame in aligned_assets.items():
        leg_trades = trades_df.loc[trades_df["asset"] == asset_key]
        leg_daily = mark_to_market_daily_returns(
            frame, leg_trades, cost_bps=float(cost_bps_by_asset[asset_key]))
        strategy_daily = (
            leg_daily if strategy_daily is None
            else (1.0 + strategy_daily) * (1.0 + leg_daily) - 1.0)

    return {
        "strategy_daily": strategy_daily,
        "in_market": in_market,
        "trades_df": trades_df,
    }


def build_pooled_curve(per_ticker_signal_fn: Mapping[str, SignalFn],
                        full_frames: Mapping[str, pd.DataFrame], *,
                        cost_bps_by_symbol: Mapping[str, float],
                        selected_count: Optional[int] = None) -> dict:
    """Account curve for the `POOLED_SHAPES` family (`pooled_single_asset`,
    `event_pooled`) -- many traded instruments under ONE hypothesis, each
    run over its OWN full frame via `build_single_asset_curve` (never
    reimplemented here), then combined into one account-level curve.

    `per_ticker_signal_fn` maps each pool member's ticker to its OWN signal
    function -- NOT one function shared across the whole pool -- matching the
    production wiring this driver's caller (scripts/run_abi1_mining.py) must
    match: `run_batch5_mining.compute_candidate`'s own per-ticker backtest
    loop builds a genuinely distinct `SignalFn` per member via
    `module.make_signal_fn(ticker, index, frames)` (through
    `run_batch5_mining._signal_factory`). A pooled family's signal can be
    cross-sectional (closing over every member's own price data to rank
    one ticker against the rest), so a single shared `SignalFn` would not
    be correct here.

    `selected_count` explicitly opts a synchronized top-N selection into
    full allocation at each cohort's entry. Weights drift with prices until
    the common exit; there is no implicit daily rebalance. Incomplete,
    overlapping or asynchronous cohorts fail closed. The default is the
    fixed-sleeve convention described below.

    Default combination, three pieces:

    - `in_market` = OR across every member's own `build_single_asset_curve`
      `in_market` mask, each reindexed onto the pool's UNION calendar with
      `fill_value=False` before the OR -- a member with no bar that session
      contributes nothing, never a spurious True.
    - `strategy_daily` = equal-weight MEAN of every member's own
      `strategy_daily` sleeve on that same union calendar, `0.0`-filled for
      a member absent that session -- via this module's `_aligned(...,
      how="mean")`, a verbatim copy of `run_batch5_mining._aligned`, the
      SAME aggregation `build_directional_panel`'s report-only panel uses.
    - `trades_df` = `pooling.pool_trades({symbol: that member's own
      trades_df})` -- every member's OWN undiluted trades, concatenated and
      ticker-tagged. The ledger row (`record_abi1_outcome`) needs a real `trades_df` shape
      regardless of tax policy (e.g. to report a trade count), and matching
      `strategy_daily`'s own pooled calendar construction keeps the two
      from ever disagreeing (a KeyError risk downstream if they diverged)
      -- correct regardless of which `tax_policy` a caller later selects.

    JUDGMENT CALL, flagged as such (same convention as this codebase's
    other invented numbers/conventions). The full argument:

    `account_relative.evaluate()` needs exactly ONE `in_market` bool mask
    and ONE `strategy_daily` series per family -- a pooled family has no
    single, unambiguous answer to "is the ACCOUNT in the market today" when
    only some members hold a position. OR-of-masks + mean-with-zero-fill is
    the CONSERVATIVE default: on a session where only 2 of 11 pool members
    are invested, the account is scored as "in the market" earning the
    DILUTED mean-of-11 return (mostly zero) rather than the cash rate on
    the 9/11 unfielded fraction. This can only UNDERSTATE a pooled
    candidate's return relative to a fractional cash-blend alternative,
    never overstate it: crediting an idle member's own slice 0.0 (this
    function's convention) instead of a non-negative cash rate (what a
    blend alternative would credit) can only pull the pooled
    `strategy_daily` DOWN relative to that alternative on any session some
    members are idle. Non-negative by construction: `evaluate_family`
    calls `account_relative.sweep_cash_daily` unconditionally
    for every family in v1 (no catalog family declares the
    alternative `parked_cash_daily` convention, which -- unlike
    `sweep_cash_daily` -- can print negative, since its underlying ^IRX
    series does on 7 real historical sessions per that function's own
    docstring, so this argument would NOT hold for it) --
    `sweep_cash_daily` is `IDLE_SWEEP_APY * elapsed_calendar_days /
    days_per_year`, a fixed positive constant times a non-negative day
    count, always >= 0. A fractional cash-blend alternative --
    crediting each idle member's own share the cash rate instead of 0.0,
    i.e. `strategy_daily = mean(invested members' sleeves) * (fraction
    invested) + cash_rate * (fraction idle)` -- was considered and is NOT
    built here: it needs a NEW convention for what "cash rate" means inside
    a pooled, fractional-account context (the existing cash producers are
    whole-account conventions, not built to credit a fractional slice of
    one session), is new quant design work of the same kind as the tax
    deferral below, and is out of scope for v1.

    Why tax is DEFERRED here, not approximated:
    `account_relative.netted_tax_factors`'s dollar-gain
    identity (`entry_equity = W / (1+net)`) is exact only when the
    account's entire compounded growth over a trade's span equals `(1+net)`
    of THAT ONE trade -- true for a single-instrument or rotation family
    (100% of capital, one position at a time), FALSE for a pooled family,
    where the account's real growth over any one member's trade span is
    the equal-weight MEAN across all N members, not that one trade in
    isolation. Feeding a pooled family's true per-member `net_ret` values
    (undiluted, available via this function's own `trades_df`) alongside
    the pool's diluted `strategy_daily` into `netted_tax_factors` would
    misattribute each trade's dollar gain/loss by a factor on the order of
    N (and a large pool can hold hundreds of names) --
    and the direction is NOT reliably conservative (it can inflate the
    modelled tax bill on a real gain OR inflate the sheltering credit on a
    real loss, depending on each trade's sign), so approximating it could
    manufacture either a false cut or a false survivor. A correct pooled
    tax model needs new machinery (e.g. sizing each trade's dollar impact
    at `equity/N` directly, rather than inverting compounding on the
    diluted curve) that is real, new quant design work -- out of scope for
    v1. Settled default (this function does NOT enforce it -- the caller must
    pass `tax_policy="none"` into `evaluate_family` for every
    pooled/event_pooled family; see `evaluate_family`): pooled/event_pooled families
    score UNTAXED in this v1, a known, disclosed simplification recorded in
    every pooled family's ledger `note` (`record_abi1_outcome`) -- not a claim that a
    pooled family's true after-tax return is being measured.
    """
    members = list(per_ticker_signal_fn.keys())
    if not members:
        raise ValueError(
            "build_pooled_curve: per_ticker_signal_fn is empty -- no pool "
            "members were passed in at all")

    per_member = {
        symbol: build_single_asset_curve(
            full_frames[symbol], per_ticker_signal_fn[symbol],
            cost_bps=float(cost_bps_by_symbol[symbol]))
        for symbol in members
    }

    # Uses the SAME _aligned union-calendar/reindex/zero-fill path for both
    # outputs rather than a second,
    # independently-built union loop for in_market -- a member's in_market
    # mask summed as 0.0/1.0 across the union calendar is > 0 exactly on a
    # session at least one member is in (the OR of the masks), and it is
    # guaranteed to share in_market's and strategy_daily's union calendar
    # by construction (never able to diverge).
    in_market = _aligned(
        {symbol: curve["in_market"].astype(float) for symbol, curve in per_member.items()},
        how="sum") > 0
    in_market.name = "in_market"

    strategy_daily = _aligned(
        {symbol: curve["strategy_daily"] for symbol, curve in per_member.items()},
        how="mean")

    trades_df = pool_trades(
        {symbol: curve["trades_df"] for symbol, curve in per_member.items()})
    if selected_count is not None:
        strategy_daily = _selected_cohort_returns(per_member, trades_df,
                                                   strategy_daily.index,
                                                   selected_count)

    return {
        "strategy_daily": strategy_daily,
        "in_market": in_market,
        "trades_df": trades_df,
    }


def _selected_cohort_returns(per_member, trades, index, selected_count):
    """Equal entry capital across N selected names, held until cohort exit."""
    if (isinstance(selected_count, bool) or not isinstance(selected_count, int)
            or not 1 <= selected_count <= len(per_member)):
        raise ValueError("selected_count must be a positive integer within the pool")
    daily = pd.Series(0.0, index=index)
    previous_exit = None
    for entry, cohort in trades.groupby("entry_date", sort=True):
        if (len(cohort) != selected_count or cohort["ticker"].nunique() != selected_count
                or cohort["exit_date"].nunique() != 1):
            raise ValueError("top-N allocation requires complete synchronized cohorts")
        end = cohort["exit_date"].iloc[0]
        if previous_exit is not None and entry <= previous_exit:
            raise ValueError("top-N allocation cannot overlap cohorts")
        span = index[(index >= entry) & (index <= end)]
        wealth = pd.Series(0.0, index=span)
        for ticker in cohort["ticker"]:
            curve = per_member[ticker]
            returns = curve["strategy_daily"].reindex(span)
            if returns.isna().any() or not curve["in_market"].reindex(span, fill_value=False).all():
                raise ValueError("top-N member is missing held sessions")
            wealth += (1.0 + returns).cumprod() / selected_count
        prior = wealth.shift(1, fill_value=1.0)
        if (prior <= 0).any():
            raise ValueError("top-N cohort exhausted its capital")
        daily.loc[span] = wealth / prior - 1.0
        previous_exit = end
    return daily


def evaluate_family(curve: dict, *, benchmark_frame: pd.DataFrame,
                     benchmark_entry_cost_bps: float, account_type: str,
                     tax_policy: str) -> dict:
    """Shape-agnostic ABI-1 assembly step -- the function every one
    of this module's curve builders (`build_single_asset_curve`,
    `build_rotation_curve`, `build_pooled_curve`) feeds into, since all
    three return the SAME `{"strategy_daily", "in_market", "trades_df"}`
    dict shape. Assembles the three pieces `account_relative.evaluate()`
    needs beyond the curve itself, and returns its full result dict
    unmodified.

    `cash_daily` -- `account_relative.sweep_cash_daily(curve["strategy_daily"].index)`,
    unconditionally. This is the ONLY cash convention this driver uses in
    v1 -- no catalog family declares the `parked_cash_daily`
    alternative -- so the ledger writer (`record_abi1_outcome`) must record the literal string
    `"sweep_cash_daily"` as `cash_convention` alongside every score,
    satisfying the requirement that the convention be pre-registered and recorded
    with the run, without building a declaration mechanism nothing yet
    uses. `build_pooled_curve`'s own conservative-direction argument
    depends on this staying unconditional --
    `parked_cash_daily` can print negative, `sweep_cash_daily` cannot.

    `benchmark_daily` -- `benchmark_relative.cost_adjusted_benchmark_daily_returns`
    run over `benchmark_frame`. `benchmark_entry_cost_bps` MUST already be
    HALVED by the caller (that function's own docstring: "Callers pass
    `entry_cost_bps` already halved") -- this function does not halve it a
    second time. `benchmark_frame` should cover the SAME full span as
    `curve["strategy_daily"]`'s index (e.g. SPY fetched with `start="max"`);
    a shorter benchmark truncates via `evaluate()`'s own inner join in
    `log_excess` and is reported, not hidden, at
    `result["window"]["benchmark_truncated"]` -- passing the full SPY
    history even for a family trading back to the 1960s is correct, not a
    bug to work around.

    Tax rates -- `tax_policy` is a REQUIRED, explicit argument, never
    hardcoded. The caller (scripts/run_abi1_mining.py) must pass
    `"realization"` for single/rotation shapes, where 100% of capital is in
    one position at a time and `netted_tax_factors`'s dollar-gain identity
    holds, and `"none"` for pooled/event_pooled shapes, where it does not
    (see `build_pooled_curve`'s own docstring: pooled tax netting is
    deferred, not approximated, in v1). Under `"realization"`,
    `account_type` -- the family's own `catalog_entry["account_types"][0]`,
    the SAME field `run_batch5_mining.compute_candidate` reads -- is fed through
    `cost_tax_inputs.tax_haircut_rate_for` exactly as that call site does:
    0.0 for a tax-advantaged account, `constants.STCG_TAX_RATE` for
    `"taxable"`, ValueError for anything else (a catalog data-entry bug,
    not a candidate this function should silently score at some other
    rate). The long-term rate is always `constants.LTCG_TAX_RATE` under
    `"realization"`. Under `"none"` both rates are 0.0 regardless of
    `account_type` -- nothing is charged, matching a `trades_df` whose
    `net_ret` is already understood to be pre-tax-and-untaxed for this call.

    Always scores at `constants.ABI1_N_TRIALS` / `constants.DAILY_SR_VARIANCE`
    -- the daily-construction pair, fixed together --
    never the per-trade `sharpe_in_sample`/`pooled_sr_variance` pool
    (`account_relative`'s own module docstring: the two are in different
    units, and mixing them moves the DSR bar's dominant term substantially).

    Raises ValueError up front if `tax_policy` is neither `"realization"`
    nor `"none"` -- `account_relative.evaluate` raises the identical error
    itself, but only after this function would already have built the cash
    and benchmark series; checking here first is strictly cheaper and the
    message is the same either way.
    """
    if tax_policy not in ("realization", "none"):
        raise ValueError(
            f"tax_policy must be 'realization' or 'none'; got {tax_policy!r}")
    risk_account_basis = _risk_account_basis(curve)
    cash = account_relative.sweep_cash_daily(curve["strategy_daily"].index)
    benchmark_daily = benchmark_relative.cost_adjusted_benchmark_daily_returns(
        benchmark_frame, entry_cost_bps=benchmark_entry_cost_bps)
    tax_rate = (cost_tax_inputs.tax_haircut_rate_for(account_type)
               if tax_policy == "realization" else 0.0)
    ltcg_rate = constants.LTCG_TAX_RATE if tax_policy == "realization" else 0.0
    return account_relative.evaluate(
        strategy_daily=curve["strategy_daily"], in_market=curve["in_market"],
        benchmark_daily=benchmark_daily, cash_daily=cash,
        n_trials=constants.ABI1_N_TRIALS, sr_variance=constants.DAILY_SR_VARIANCE,
        trades_df=curve["trades_df"], tax_policy=tax_policy,
        tax_haircut_rate=tax_rate, ltcg_tax_rate=ltcg_rate,
        risk_account_basis=risk_account_basis)


def _risk_account_basis(curve: dict) -> str:
    """The account `account_relative.risk_refusals` measures for this curve.

    A curve that declares
    ``account_kind == "benchmark_plus_overlay"`` (SPY held at all times plus an
    overlay; `long_overlay` and `beta_neutral_overlay` set it) is judged on the
    OVERLAY-ONLY EXCESS account. A curve with no ``account_kind`` key -- every
    other builder, and every hand-built dict -- is judged on the whole
    account. Any other value is refused rather than guessed at: a
    typo here would silently pick the harsher whole-account basis.
    """
    kind = curve.get("account_kind")
    if kind is None:
        return "whole_account"
    if kind == account_relative.ACCOUNT_KIND_BENCHMARK_PLUS_OVERLAY:
        return "overlay_only_excess"
    raise ValueError(
        f"curve['account_kind'] must be absent or "
        f"{account_relative.ACCOUNT_KIND_BENCHMARK_PLUS_OVERLAY!r}; got {kind!r}")


def record_abi1_outcome(ledger_path: str, family_id: str, run_id: str,
                        tested_date: str, result: dict, *,
                        catalog_entry: dict, instrument_set,
                        cash_convention: str, benchmark: str,
                        tax_policy: str, ticker_label: Optional[str] = None,
                        signal_source_path: str,
                        signal_source_sha256: str,
                        extra_note: Optional[str] = None,
                        unscored_note: Optional[str] = None) -> str:
    """Maps ONE `account_relative.evaluate()` result dict (`evaluate_family`,
    or a hand-built dict of the same shape) onto exactly
    the write `trial_ledger.record_outcome` needs.
    Returns the `stage` string actually written -- the four SCORED verdicts
    write their own name (`"survived"`/`"cut_dsr"`/`"cut_holdout"`/
    `"cut_risk_refusal"`); both UNSCORED verdicts (`"not_scoreable"`/
    `"holdout_not_evaluable"`) write `"insufficient_evidence"` -- so a
    caller's own run report can log what happened without re-deriving it
    from `result["verdict"]`.

    Two safeguards are baked in here rather than left to the caller:

    1. `family_id` is used VERBATIM, never derived from `result` or from a
       pool's own display label. The caller (scripts/run_abi1_mining.py) is the one place that
       knows the catalog's real permanent id (`catalog_entry["family_id"]`)
       and MUST pass it here -- NEVER
       `run_batch5_mining.pool_ticker_label`'s report-safe collapsed string
       (which degenerates to a generic `"pool-of-<N>"` label for a large
       pool). Passing that instead would break every ledger
       accessor keyed on the real family_id, silently MERGE two unrelated
       large pools that happen to share a member count onto the identical
       fake id, and defeat the caller's own re-run guard (which reads the real
       family_id and would never find a wrongly-keyed row, so every re-run
       would re-score and duplicate the outcome). `ticker_label` is the
       SEPARATE, purpose-built field for a pool's display name
       (`record_outcome`'s own parameter) and is simply
       forwarded through unchanged -- `None` by default, exactly matching
       `record_outcome` itself.

    2. `holdout_not_evaluable`'s evidence payload needs a REAL pinned
       holdout window: `evidence.build_abi1_holdout_not_evaluable`'s
       `pinned_window` is a required dict, not `Optional` -- there is no
       safe `None` fallback for this reason code, on purpose (a caller must
       always know which sessions it is claiming could not be evaluated).
       Derived here from `result["account_daily"].index` -- the full scored
       account curve `evaluate()` always returns, real even when the
       holdout gate itself never ran:
         * `result["window"]["holdout_split_applied"] is True` -- a real
           holdout sub-window existed (`holdout_gate` itself is what
           refused, e.g. `non_finite_excess`/`zero_variance`) -- pin the
           TAIL `result["window"]["holdout_sessions"]` dates, i.e. exactly
           the sessions `split_holdout` carved off and `holdout_gate` was
           actually handed.
         * `False` -- `split_holdout` itself refused (`holdout_too_short`/
           `in_sample_too_short`; no holdout sub-window exists at all) --
           pin the FULL account span instead, the same span
           `account_start`/`account_end` below already use.

    `account_start`/`account_end` (both UNSCORED reason codes) are always
    `result["account_daily"].index[0]`/`[-1]` -- the family's full scored
    span, real even when it is short. This is a SEPARATE field from
    `holdout_not_evaluable`'s own `pinned_window` above; the two coincide
    only when the holdout split itself was refused (bullet 2's second
    case).

    For the four SCORED stages, `note` carries `cash_convention`,
    `benchmark`, AND `tax_policy` in one literal template -- reusing the
    existing free-text field rather than a schema change. For a pooled
    family scored under `tax_policy="none"`, this note IS
    the disclosure that no tax was modeled for that row; it is written for
    EVERY scored row, never suppressed for a pooled one. The two UNSCORED
    rows do not duplicate this into `note` -- both evidence builders
    already carry `cash_convention`/`benchmark` inside their own `observed`
    block, and `tax_policy` is not a dimension either UNSCORED reason code
    models (a family that never reached a real score has no tax-relevant
    disclosure to make).

    `extra_note` (optional; e.g. the bespoke `scripts/
    run_cboe_rotation_mining.py` driver passes one): appended verbatim to the
    SCORED stage's `note`, `"; "`-joined after `tax_policy`. That driver
    prices one leg off a DISCLOSED ASSUMED constant rather than a
    real captured quote (the underlying instrument is a non-tradable
    calculated index) -- without this, the permanent ledger row would look
    identical to any ordinary real-quote-priced family, and "a recorded
    trial must be priced from a measured quote sample" (this repo's own
    standing principle, `run_batch5_mining.refuse_synthetic_quotes`) would
    be silently violated in spirit even though no literal quote was
    fabricated. `None` (default) writes the plain template above.

    `unscored_note` (optional, for the opt-in history-end
    setting): written verbatim as the `note` of the two
    UNSCORED rows, which otherwise carry none. An unscored run still spends
    the family's one permanent ABI-1 slot (the `construction == "ABI-1"`
    re-run guard), so a setting that changes what was scored must be readable
    on that row too. It is a SEPARATE keyword, not a widening of `extra_note`:
    the many driver scripts that pass `extra_note` keep unscored rows with no
    note. `None`/empty (default) writes `note=None`; a SCORED row never
    reads it.

    `evidence_source="measured"` and a `reactivation={"kind":
    "none_available", ...}` block are HARDCODED here, not caller
    parameters: v1 has no ABI-1 reactivation mechanism at all (the caller has
    no `--retry` flag in v1 -- every prior ABI-1 score for a
    family blocks that family from re-scoring forever, via the
    `construction == "ABI-1"` re-run guard the caller applies), so every ABI-1
    evidence payload this function can ever write is a first measurement
    with nothing yet to react to. A reactivation mechanism would need its
    own parameters here, not a silently-reused default.

    Raises ValueError if `result["verdict"]` is not one of
    `account_relative.VERDICTS` -- a caller passing something else is a
    programming error this function should not paper over with a guess.

    Raises `trial_ledger.OutcomeAlreadyRecorded` if a
    concurrent caller's outcome for this exact family landed first, between
    this caller's own early already_scored prefilter and this function's
    write -- the freshly computed `result` is discarded, not retried; see
    `trial_ledger.record_outcome_if_not_scored`'s own docstring for the
    full race this closes.
    """
    verdict = result["verdict"]
    if verdict not in account_relative.VERDICTS:
        raise ValueError(
            f"unknown ABI-1 verdict {verdict!r}; expected one of "
            f"{account_relative.VERDICTS}")

    # Register the performance-bearing attempt through the same idempotent
    # API as the per-trade runner. Use the scored calendar, which can differ
    # from account history when the benchmark starts later. A refused split
    # has no holdout: null boundaries disclose that rather than invent dates.
    scored_index = result.get("log_excess", result["account_daily"]).index
    holdout_start = holdout_end = None
    if result["window"]["holdout_split_applied"]:
        n_holdout = result["window"]["holdout_sessions"]
        if not 1 <= n_holdout <= len(scored_index):
            raise ValueError("holdout_sessions must fit the scored calendar")
        holdout_start = scored_index[-n_holdout].strftime("%Y-%m-%d")
        holdout_end = scored_index[-1].strftime("%Y-%m-%d")

    def register_first_test():
        # Empty representatives match the per-trade registration loop; daily
        # log excess must not be mixed into its per-trade correlation pool.
        trial_ledger.record_first_test(
            ledger_path, family_id, tested_date, holdout_start, holdout_end,
            [], run_id)

    if verdict in evidence.SCORED_STAGES:
        if result["layer1"]["dsr_value"] is None or not math.isfinite(result["layer1"]["dsr_value"]):
            raise ValueError("a scored outcome requires a finite dsr_value")
        note = (f"cash_convention={cash_convention}; "
               f"benchmark={benchmark}; tax_policy={tax_policy}")
        if extra_note:
            note += f"; {extra_note}"
        # Outcome write FIRST, first_test registration second:
        # record_outcome is the chokepoint that validates the
        # stage/evidence shape and can raise; if it raised AFTER
        # register_first_test() already succeeded, this family's holdout
        # window would be permanently pinned in the ledger with no
        # corresponding outcome -- a phantom "tested" row that inflates
        # cumulative_family_count without a real result behind it. Ordering
        # the calls this way means a record_outcome failure leaves no trace,
        # and the (simpler, less error-prone) first_test registration is the
        # one call whose own failure just needs a retry, not a repair.
        #
        # record_outcome_if_not_scored, not record_outcome:
        # closes the TOCTOU between a caller's
        # own early unlocked already_scored prefilter and this write -- two
        # concurrent callers could otherwise both reach here and both
        # append a permanent outcome for the same family. A False return
        # means this call lost that race; raise so the caller (which has
        # already spent its compute) discards the result instead of
        # silently reporting a write that never happened.
        recorded = trial_ledger.record_outcome_if_not_scored(
            ledger_path, family_id, run_id, tested_date, stage=verdict,
            dsr_value=result["layer1"]["dsr_value"],
            sharpe_in_sample=result["layer1"]["sharpe_daily_excess"],
            ticker_label=ticker_label, construction="ABI-1",
            note=note)
        if not recorded:
            raise trial_ledger.OutcomeAlreadyRecorded(
                f"{family_id!r} already carries a construction=\"ABI-1\" "
                f"outcome -- lost a race against a concurrent scoring "
                f"attempt for the same family; this freshly computed "
                f"result is discarded, not written")
        register_first_test()
        return verdict

    # The two UNSCORED verdicts share account_start/account_end and the
    # hardcoded evidence_source/reactivation block; only pinned_window
    # (holdout_not_evaluable only) and the builder itself differ.
    account_index = result["account_daily"].index
    account_start = account_index[0].strftime("%Y-%m-%d")
    account_end = account_index[-1].strftime("%Y-%m-%d")
    reactivation = {
        "kind": "none_available",
        "rationale": (
            "ABI-1 v1 has no reactivation mechanism -- every family this "
            "construction has ever scored is permanently blocked from "
            "re-scoring by the construction=\"ABI-1\" re-run guard, so "
            "there is nothing to react to yet."),
    }

    if verdict == "not_scoreable":
        payload = evidence.build_abi1_not_scoreable(
            n_sessions=result["layer1"]["n_sessions"],
            abi1_reason=result["layer1"]["reason"],
            catalog_entry=catalog_entry, instrument_set=instrument_set,
            account_start=account_start, account_end=account_end,
            cash_convention=cash_convention, benchmark=benchmark,
            signal_source_path=signal_source_path,
            signal_source_sha256=signal_source_sha256,
            evidence_source="measured", reactivation=reactivation)
    elif verdict == "holdout_not_evaluable":
        if result["window"]["holdout_split_applied"]:
            n_holdout = result["window"]["holdout_sessions"]
            # n_holdout must be >= 1: through the real evaluate() path this
            # is guaranteed (account_relative.split_holdout raises ValueError
            # for holdout_sessions < 2 before holdout_split_applied can ever
            # become True), but this function also accepts a hand-built
            # result dict (see docstring) where nothing enforces that. A
            # non-positive n_holdout would silently make account_index[-0]
            # == account_index[0] (Python's -0 == 0), mislabeling the
            # family's ENTIRE account span as the holdout tail in an
            # append-only ledger row.
            if n_holdout < 1:
                raise ValueError(
                    f"result['window']['holdout_sessions'] must be >= 1 "
                    f"when holdout_split_applied is True; got {n_holdout}")
            pinned_window = {
                "holdout_start": account_index[-n_holdout].strftime("%Y-%m-%d"),
                "holdout_end": account_index[-1].strftime("%Y-%m-%d"),
            }
        else:
            pinned_window = {"holdout_start": account_start,
                             "holdout_end": account_end}
        payload = evidence.build_abi1_holdout_not_evaluable(
            in_sample_dsr_value=result["layer1"]["dsr_value"],
            in_sample_sharpe_daily_excess=result["layer1"]["sharpe_daily_excess"],
            n_in_sample_sessions=result["window"]["in_sample_sessions"],
            n_holdout_sessions=result["holdout"]["n_holdout"],
            holdout_sessions_required=result["holdout"]["required_sessions"],
            holdout_refusal=result["holdout"]["refusal"],
            catalog_entry=catalog_entry, instrument_set=instrument_set,
            account_start=account_start, account_end=account_end,
            pinned_window=pinned_window,
            cash_convention=cash_convention, benchmark=benchmark,
            signal_source_path=signal_source_path,
            signal_source_sha256=signal_source_sha256,
            evidence_source="measured", reactivation=reactivation)
    else:
        # Reached only if account_relative.VERDICTS ever grows a 7th member
        # that isn't in evidence.SCORED_STAGES and isn't "not_scoreable" --
        # fail loudly rather than silently mis-processing an unrecognized
        # verdict as holdout_not_evaluable (a bare `else:` standing for
        # "holdout_not_evaluable" would do exactly that, since the
        # top-of-function VERDICTS check only catches an unknown STRING,
        # not a valid-but-unhandled one).
        raise AssertionError(
            f"ABI-1 verdict {verdict!r} is in account_relative.VERDICTS but "
            f"unhandled by record_abi1_outcome -- this function needs a new "
            f"branch, not a guess")

    evidence.validate(payload, stage="insufficient_evidence", at_write_time=True)
    # Same ordering as the SCORED branch above (outcome write first), and the
    # same race-closing record_outcome_if_not_scored call.
    recorded = trial_ledger.record_outcome_if_not_scored(
        ledger_path, family_id, run_id, tested_date,
        stage="insufficient_evidence", evidence=payload,
        ticker_label=ticker_label, construction="ABI-1",
        note=unscored_note or None)
    if not recorded:
        raise trial_ledger.OutcomeAlreadyRecorded(
            f"{family_id!r} already carries a construction=\"ABI-1\" "
            f"outcome -- lost a race against a concurrent scoring attempt "
            f"for the same family; this freshly computed result is "
            f"discarded, not written")
    register_first_test()
    return "insufficient_evidence"
