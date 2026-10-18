"""Rough-backtest driver for margin_debt_deleverage_synthetic_leverage_flip
(the cost-input and rough-backtest stages).

Scope: a family's test runs in four stages -- (1) a duplicate check,
(2) cost inputs, (3) a rough backtest, (4) ABI-1 scoring plus the ledger
write. The duplicate check has no code artifact; the related
margin_debt_growth_regime family is distinguished in this family's own
module docstring. This driver runs stages 2 and 3 (the rough backtest
reports in-sample data only); stage 4 is
`scripts/run_margin_debt_deleverage_abi1.py`, which prices the curve from a
dated quotes file. `prepare_abi1_scoring_call` below returns a
zero-argument callable that runs `abi1_driver.evaluate_family` on this
driver's curve (the same call stage 4 makes on its own cost-sampled
curve); `main()` builds that callable as a wiring check and does not call
it. This driver does not call `abi1_driver.record_abi1_outcome`, writes
nothing to `data/mining/trial_ledger.jsonl`, and does not touch
`catalog.json` or any `accepted.json`: a rough backtest records nothing.

Cost: this driver reads no quotes, so every run prices both legs at
``cost_bps=0.0`` (the rough backtest is GROSS of trading cost) and the
output JSON's ``cost_model`` says so rather than silently defaulting. Under
the pipeline's rule that a cost number is never invented or estimated, a
usable cost sample is one captured DURING NYSE regular trading hours
(09:30-16:00 ET) and stamped ``market_state`` (batch5's own quote-capture
convention). `market_is_open_now` (a weekday clock check) selects the
``cost_model`` reason text: outside regular hours the reason states that
the cost sample is skipped, because a quote taken then cannot be certified
as a genuinely executable regular-session spread (the quote source gives no
session-state field).

Builds the curve with the shared ABI-1 v1 primitives
(``mining.abi1_driver.build_rotation_curve`` / ``evaluate_family``), as
the other bespoke drivers do (``run_margin_debt_mining.py``,
``run_synthetic_leveraged_dip_mining.py``).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
SCRIPTS_ROOT = os.path.join(REPO_ROOT, "scripts")
for _path in (SRC_ROOT, SCRIPTS_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import pandas as pd

import data_pipeline
import run_batch5_mining as batch5
from mining import abi1_driver, constants, evidence, external_data
from mining.batch5 import margin_debt_deleverage_synthetic_leverage_flip as mod
from mining.batch5.synthetic_leveraged_dip_buy_pooled_largecap import (
    synthetic_ohlc_from_close)

UNDERLYING = "SPY"
DEFENSIVE = "SH"
ACCOUNT_TYPE = "taxable"  # margin-financed leverage + a short ETF are not
                          # generally meaningful in a tax-advantaged
                          # account -- same reasoning synthetic_leveraged_
                          # dip_buy_pooled_largecap.py already gives.
CASH_CONVENTION = "sweep_cash_daily"
TAX_POLICY = "realization"  # rotation_two_asset shape, matches
                            # abi1_driver.evaluate_family's own documented
                            # rule.
MARKET_BENCHMARK = "SPY"
MARKET_OPEN_ET = datetime.time(9, 30)
MARKET_CLOSE_ET = datetime.time(16, 0)

DEFAULT_RUN_ID = "margin-debt-deleverage"
# The date a run is recorded as tested: today, unless --tested-date is passed.
DEFAULT_TESTED_DATE = datetime.date.today().isoformat()
RESEARCH_ROOT = batch5.RESEARCH_ROOT
PROVENANCE_PATH = os.path.join(
    REPO_ROOT, "data", "mining", "margin_debt_deleverage", "provenance.json")

_SIGNAL_SOURCE_PATH = os.path.relpath(__file__, REPO_ROOT).replace(os.sep, "/")


# ------------------------------------------------------------- cost inputs

def market_is_open_now() -> tuple:
    """(is_open, reason). Uses a real US/Eastern wall-clock read (no
    network) -- NYSE weekday regular-session hours only; does not account
    for market holidays (a holiday reads as "open" by this check alone).
    The result only selects the ``cost_model`` reason text in ``main()``;
    the rough backtest runs gross of cost either way (see module
    docstring)."""
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        now_et = datetime.datetime.utcnow()
    if now_et.weekday() >= 5:
        return False, f"weekend ({now_et.strftime('%A')} {now_et.isoformat()})"
    if not (MARKET_OPEN_ET <= now_et.time() <= MARKET_CLOSE_ET):
        return False, (
            f"outside regular trading hours ({now_et.strftime('%H:%M:%S %Z') or now_et.isoformat()} "
            f"ET; regular session is {MARKET_OPEN_ET}-{MARKET_CLOSE_ET} ET)")
    return True, "within regular trading hours (weekday clock check only)"


# ------------------------------------------------------------- real data

def _fetch_spy_ohlc() -> pd.DataFrame:
    frame, _origin = batch5.load_frame(
        UNDERLYING, ohlc_loader=batch5.default_ohlc_loader,
        cache_dir=os.path.join(RESEARCH_ROOT, DEFAULT_RUN_ID, "artifacts", "ohlc"))
    return frame


def _fetch_sh_ohlc() -> pd.DataFrame:
    frame, _origin = batch5.load_frame(
        DEFENSIVE, ohlc_loader=batch5.default_ohlc_loader,
        cache_dir=os.path.join(RESEARCH_ROOT, DEFAULT_RUN_ID, "artifacts", "ohlc"))
    return frame


def _fetch_irx() -> pd.Series:
    return data_pipeline.fetch_risk_free_rate(start="max")


def fetch_margin_debt_series():
    """Real (or injected-for-test) fetch of the FINRA workbook. Returns
    ``(series, metadata)`` -- never a bare series, so provenance is always
    available to the caller. Same convention as
    ``run_margin_debt_mining.fetch_margin_debt_series``."""
    snapshot = external_data.fetch_finra_margin_debt()
    return snapshot.series, snapshot.metadata


def build_synthetic_frame(spy_frame: pd.DataFrame, irx_series: pd.Series):
    """SPY's synthetic leveraged-long frame, built exactly the way
    ``synthetic_leveraged_dip_buy_pooled_largecap``'s own bespoke driver
    builds one per pool member -- reused, not reimplemented.
    ``build_synthetic_leveraged`` needs a ``Date`` COLUMN plus a column
    literally named ``UNDERLYING`` (``build_synthetic_3x``'s own calling
    convention); this adapts the shape at the call site. Passes
    leverage/annual_expense_ratio/financing_spread EXPLICITLY from the
    family module's constants (derived from mod.PARAMETERS), so they are
    drift-protected by parameter_drift rather than left to the function's
    own defaults -- the same convention run_post_opex_gamma_release_abi1.py
    follows."""
    df_for_synth = pd.DataFrame({
        "Date": spy_frame.index, UNDERLYING: spy_frame["Close"].to_numpy(),
    })
    synth = data_pipeline.build_synthetic_leveraged(
        df_for_synth, UNDERLYING, annual_financing_rate=irx_series,
        leverage=mod.LEVERAGE, annual_expense_ratio=mod.ANNUAL_EXPENSE_RATIO,
        financing_spread=mod.FINANCING_SPREAD)
    synthetic_id = f"{UNDERLYING}_{int(mod.LEVERAGE)}x_synthetic"
    close = synth[synthetic_id].to_numpy()
    frame = synthetic_ohlc_from_close(spy_frame.index, close, name=synthetic_id)
    return frame, synthetic_id


# --------------------------------------------------------- rough backtest

def in_sample_split(index: pd.DatetimeIndex, *,
                    holdout_months: int = constants.HOLDOUT_WINDOW_MONTHS
                    ) -> pd.DatetimeIndex:
    """Everything strictly before the trailing ``holdout_months``-month
    window (the rough backtest reads in-sample data only: everything before
    the holdout window). Uses the pipeline's own pre-registered
    ``constants.HOLDOUT_WINDOW_MONTHS`` default (6) -- this is a
    PROVISIONAL split for the rough backtest only; the holdout that counts
    is the one the ABI-1 scoring stage applies and records, and this
    driver never touches the trial ledger."""
    if len(index) == 0:
        return index
    cutoff = index.max() - pd.DateOffset(months=holdout_months)
    return index[index <= cutoff]


def descriptive_stats(daily_returns: pd.Series, trades_df: pd.DataFrame) -> dict:
    """Plain descriptive stats over the in-sample slice -- deliberately
    NOT the Deflated Sharpe Ratio (ABI-1 scoring's job, not run here).
    Never raises on zero trades / zero-variance input."""
    n_sessions = int(len(daily_returns))
    n_trades = int(len(trades_df))
    values = daily_returns.to_numpy(dtype=float) if n_sessions else np.array([])
    cumulative = float(np.prod(1.0 + values) - 1.0) if n_sessions else 0.0
    n_years = n_sessions / 252.0 if n_sessions else 0.0
    if n_years > 0 and (1.0 + cumulative) > 0:
        cagr = float((1.0 + cumulative) ** (1.0 / n_years) - 1.0)
    else:
        cagr = None
    ann_vol = float(np.std(values, ddof=1) * np.sqrt(252.0)) if n_sessions > 1 else None
    if ann_vol not in (None, 0.0):
        sharpe = float((np.mean(values) * 252.0) / ann_vol)
    else:
        sharpe = None
    if n_sessions:
        equity = np.cumprod(1.0 + values)
        running_max = np.maximum.accumulate(equity)
        drawdown = equity / running_max - 1.0
        max_dd = float(np.min(drawdown))
    else:
        max_dd = None
    win_rate = (float((trades_df["net_ret"] > 0).mean())
               if n_trades else None)
    avg_trade_net_ret = (float(trades_df["net_ret"].mean())
                         if n_trades else None)
    return {
        "n_sessions": n_sessions, "n_trades": n_trades,
        "cumulative_net_return": cumulative, "cagr": cagr,
        "annualized_vol": ann_vol, "sharpe_daily": sharpe,
        "max_drawdown": max_dd, "win_rate": win_rate,
        "avg_trade_net_ret": avg_trade_net_ret,
    }


def rough_backtest(*, spy_frame: pd.DataFrame, sh_frame: pd.DataFrame,
                   margin_debt: pd.Series, irx_series: pd.Series,
                   cost_bps_by_asset: dict,
                   holdout_months: int = constants.HOLDOUT_WINDOW_MONTHS) -> dict:
    """The rough backtest: run the mechanism through the real generic engine
    (``mining.rotation_engine.run_two_asset_rotation`` via
    ``abi1_driver.build_rotation_curve``, never a bespoke re-implementation)
    over the FULL history, then restrict to in-sample only for the
    reported stats. Returns the full curve too (``prepare_abi1_scoring_call``
    wraps it)."""
    synthetic_frame, synthetic_id = build_synthetic_frame(spy_frame, irx_series)
    asset_frames = {synthetic_id: synthetic_frame, DEFENSIVE: sh_frame}
    target_fn = mod.make_target_fn(spy_frame.index, margin_signal=margin_debt)
    curve = abi1_driver.build_rotation_curve(
        asset_frames, target_fn,
        cost_bps_by_asset={
            synthetic_id: float(cost_bps_by_asset[UNDERLYING]),
            DEFENSIVE: float(cost_bps_by_asset[DEFENSIVE]),
        })

    full_index = curve["strategy_daily"].index
    in_sample_index = in_sample_split(full_index, holdout_months=holdout_months)
    trades_df = curve["trades_df"]
    in_sample_trades = (
        trades_df.loc[trades_df["exit_date"] <= in_sample_index.max()]
        if len(in_sample_index) else trades_df.iloc[0:0])
    strategy_daily_in_sample = curve["strategy_daily"].loc[in_sample_index]
    stats = descriptive_stats(strategy_daily_in_sample, in_sample_trades)
    return {
        "curve": curve, "synthetic_id": synthetic_id,
        "in_sample_index": in_sample_index, "in_sample_trades": in_sample_trades,
        "stats": stats,
    }


# ------------------------------- ABI-1 scoring callable (a wiring check in main)

def prepare_abi1_scoring_call(curve: dict, *, benchmark_frame: pd.DataFrame,
                              benchmark_entry_cost_bps: float,
                              account_type: str, tax_policy: str):
    """Returns a zero-argument callable that runs ABI-1 scoring on the
    rough curve -- the same ``abi1_driver.evaluate_family`` call every
    bespoke ABI-1 driver (``run_margin_debt_mining.py``,
    ``run_synthetic_leveraged_dip_mining.py``) makes on its way to
    ``record_abi1_outcome``. Building the callable does not run it.
    `main()` builds it, asserts it is callable, and does not call it, so
    this driver computes no score."""
    def _run():
        return abi1_driver.evaluate_family(
            curve, benchmark_frame=benchmark_frame,
            benchmark_entry_cost_bps=benchmark_entry_cost_bps,
            account_type=account_type, tax_policy=tax_policy)
    return _run


# --------------------------------------------------------------- CLI

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--tested-date", default=DEFAULT_TESTED_DATE)
    parser.add_argument("--research-root", default=RESEARCH_ROOT)
    parser.add_argument("--holdout-months", type=int,
                        default=constants.HOLDOUT_WINDOW_MONTHS)
    args = parser.parse_args(argv)

    spy_frame = _fetch_spy_ohlc()
    sh_frame = _fetch_sh_ohlc()
    irx_series = _fetch_irx()
    margin_debt, finra_metadata = fetch_margin_debt_series()

    os.makedirs(os.path.dirname(PROVENANCE_PATH), exist_ok=True)
    with open(PROVENANCE_PATH, "w", encoding="utf-8") as handle:
        json.dump({"fetched_for_run_id": args.run_id, **finra_metadata}, handle, indent=2)
        handle.write("\n")

    is_open, reason = market_is_open_now()
    cost_sample_used = False
    cost_bps_by_asset = {UNDERLYING: 0.0, DEFENSIVE: 0.0}
    cost_note = (
        f"Cost sample SKIPPED -- {reason}. Rough backtest below is "
        "GROSS of trading cost (cost_bps=0.0 both legs); no cost figure was "
        "invented or estimated. A quote taken outside market hours is NOT "
        "used as a cost basis: it cannot be certified as a "
        "genuinely executable regular-session spread."
        if not is_open else
        f"Cost sample basis: {reason}."
    )

    result = rough_backtest(
        spy_frame=spy_frame, sh_frame=sh_frame, margin_debt=margin_debt,
        irx_series=irx_series, cost_bps_by_asset=cost_bps_by_asset,
        holdout_months=args.holdout_months)

    scoring_call = prepare_abi1_scoring_call(
        result["curve"], benchmark_frame=spy_frame, benchmark_entry_cost_bps=0.0,
        account_type=ACCOUNT_TYPE, tax_policy=TAX_POLICY)
    assert callable(scoring_call)  # wiring check only; main() does not call it

    run_dir = os.path.join(args.research_root, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    payload = {
        "schema_version": 1, "step": "rough_backtest_only",
        "family_id": mod.FAMILY_ID, "run_id": args.run_id,
        "tested_date": args.tested_date,
        "driver": _SIGNAL_SOURCE_PATH,
        "driver_sha256": evidence.source_sha256(__file__),
        "shape": mod.SHAPE, "traded": mod.TRADED, "parameters": mod.PARAMETERS,
        "synthetic_id": result["synthetic_id"],
        "cost_model": {
            "cost_sample_used": cost_sample_used,
            "reason": cost_note,
            "cost_bps_by_asset": cost_bps_by_asset,
        },
        "holdout_months": args.holdout_months,
        "in_sample_stats": result["stats"],
        "in_sample_sessions": int(len(result["in_sample_index"])),
        "full_history_sessions": int(len(result["curve"]["strategy_daily"])),
        "note": (
            "This driver runs stages 2-3 (cost inputs and the rough "
            "backtest, gross of cost). ABI-1 scoring (stage 4 of the four "
            "stages in this module's docstring: "
            "abi1_driver.evaluate_family + record_abi1_outcome + "
            "trial_ledger write) is run_margin_debt_deleverage_abi1.py; "
            "main() builds the prepare_abi1_scoring_call() callable as a "
            "wiring check and does not call it."
        ),
    }
    output_path = os.path.join(run_dir, "rough-backtest.json")
    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False, default=str)
        handle.write("\n")
    print(json.dumps({"output": output_path.replace(os.sep, "/"),
                      "in_sample_stats": result["stats"]}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
