"""Rough-backtest driver (data fetch, cost input, rough backtest) for the
candidate vix1d_vix9d_inversion_deleverage_trigger. Fetches real data,
builds the synthetic leveraged leg, and runs the ROUGH backtest (output
key ``step3``) via the real generic
``mining.rotation_engine.run_two_asset_rotation`` engine on in-sample
data only.

Scope: this script ends at the rough backtest. It does not import
``mining.abi1_driver``, ``mining.account_relative`` or
``mining.trial_ledger``, computes no DSR or ABI-1 score, and writes
nothing to ``data/mining/catalog.json``, ``data/mining/trial_ledger.jsonl``
or any ``accepted.json``. ``--prepare-score-call`` adds a plain-text
description of the ABI-1 scoring interface (output key
``step4_prepared_not_executed``: module names and the
``build_rotation_curve``/``evaluate_family``/``record_abi1_outcome`` call
signatures); the script prints that text and imports or calls none of it.

Cost input (output key ``step2_cost_bps``): a single live bid/ask sample for XLK,
captured OUTSIDE this script
(this script has no quote access) and passed in via ``--xlk-bid``/
``--xlk-ask``. Both legs price off this one real quote -- see the
candidate module's own docstring for why (the leveraged leg is realized
as real margin-financed shares of the SAME real fund, not a separately
quoted product).

Vendor data (see the candidate module's own docstring): yfinance can
return only a single row for ``^VIX1D``/``^VIX9D`` regardless of call
shape, and then the rough backtest cannot get past the data-alignment
step. This script still runs the real fetch + alignment, and when a
signal ticker returns nothing, or fewer than
``MIN_SIGNAL_ROWS_TO_ATTEMPT_BACKTEST`` common sessions remain, it reports
exactly what came back and exits 1 instead of backtesting. Given full
history for both tickers, the same code runs the rough backtest.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _path in (SRC_ROOT,):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import pandas as pd  # noqa: E402

import data_pipeline  # noqa: E402
from mining import constants, cost_tax_inputs, rotation_engine  # noqa: E402
from mining.batch5 import (  # noqa: E402
    vix1d_vix9d_inversion_deleverage_trigger as mod,
)

MIN_SIGNAL_ROWS_TO_ATTEMPT_BACKTEST = 30  # the minimum number of sessions
# common to both traded legs and both signal tickers (align_to_common_calendar
# raises below it): it rejects a single-row signal response, but does not by
# itself guarantee a non-empty in-sample window after the holdout split
# (run_rough_backtest raises if that window is empty).


def fetch_traded_legs(*, irx_series: pd.Series) -> dict:
    """Real XLK OHLC plus its synthetic leveraged sibling, built via
    ``data_pipeline.build_synthetic_leveraged`` financed off the real,
    time-varying ``^IRX`` series -- matches ``scripts/run_synthetic_
    leveraged_dip_mining.py``'s own established driver precedent."""
    xlk = data_pipeline.fetch_ohlc(mod.UNDERLYING, start="max")
    if xlk is None or xlk.empty:
        raise RuntimeError(f"fetch_ohlc({mod.UNDERLYING!r}) returned no data")
    xlk = xlk.set_index("Date")
    xlk.index = pd.DatetimeIndex(pd.to_datetime(xlk.index))

    df_for_synth = pd.DataFrame({
        "Date": xlk.index, mod.UNDERLYING: xlk["Close"].to_numpy(),
    })
    synth = data_pipeline.build_synthetic_leveraged(
        df_for_synth, mod.UNDERLYING, leverage=mod.LEVERAGE,
        annual_financing_rate=irx_series)
    synthetic_close = synth[mod.synthetic_id_for(mod.UNDERLYING)].to_numpy()
    synthetic_ohlc = mod.synthetic_ohlc_from_close(
        xlk.index, synthetic_close, name=mod.synthetic_id_for(mod.UNDERLYING))

    return {mod.TRADED[1]: xlk, mod.TRADED[0]: synthetic_ohlc}


def fetch_signal_legs() -> dict:
    out = {}
    for ticker in mod.SIGNAL:
        frame = data_pipeline.fetch_ohlc(ticker, start="max")
        if frame is None or frame.empty:
            out[ticker] = None
            continue
        frame = frame.set_index("Date")
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
        out[ticker] = frame
    return out


def align_to_common_calendar(asset_frames: dict, signal_frames: dict) -> tuple:
    """Trim the two traded legs to their common inception, then intersect
    the signal legs onto that calendar. Raises ValueError naming exactly
    what is missing/too short, rather than letting a bare KeyError or an
    empty-frame surprise reach the engine."""
    missing_signal = [t for t, f in signal_frames.items() if f is None or f.empty]
    if missing_signal:
        raise ValueError(
            f"no OHLC data returned for signal ticker(s): {missing_signal}")

    common_inception = max(f.index.min() for f in asset_frames.values())
    trimmed_assets = {k: f.loc[f.index >= common_inception].copy()
                       for k, f in asset_frames.items()}
    asset_calendar = None
    for frame in trimmed_assets.values():
        asset_calendar = frame.index if asset_calendar is None else asset_calendar
        if not frame.index.equals(asset_calendar):
            raise ValueError(
                "the two traded legs do not share an identical calendar "
                "after trimming to their common inception")

    common = asset_calendar
    for frame in signal_frames.values():
        common = common.intersection(frame.index)
    if len(common) < MIN_SIGNAL_ROWS_TO_ATTEMPT_BACKTEST:
        raise ValueError(
            "insufficient real signal history to attempt a backtest: the "
            f"traded calendar intersected with {list(signal_frames)} yields "
            f"only {len(common)} common session(s) (floor is "
            f"{MIN_SIGNAL_ROWS_TO_ATTEMPT_BACKTEST}) -- per-ticker raw row "
            "counts: " + ", ".join(
                f"{t}={len(f)}" for t, f in signal_frames.items()))

    aligned_assets = {k: f.loc[f.index.isin(common)].copy()
                       for k, f in trimmed_assets.items()}
    aligned_signals = {k: f.loc[f.index.isin(common)].copy()
                        for k, f in signal_frames.items()}
    return aligned_assets, aligned_signals


def split_holdout(index: pd.DatetimeIndex, *, holdout_months: int) -> tuple:
    """In-sample = every session strictly before (last session -
    holdout_months); holdout = the tail. The rough backtest uses in-sample only."""
    if len(index) == 0:
        return index, index
    cutoff = index.max() - pd.DateOffset(months=holdout_months)
    in_sample = index[index < cutoff]
    holdout = index[index >= cutoff]
    return in_sample, holdout


def run_rough_backtest(aligned_assets: dict, aligned_signals: dict, *,
                        cost_bps: float) -> dict:
    in_sample_idx, holdout_idx = split_holdout(
        aligned_assets[mod.TRADED[1]].index,
        holdout_months=constants.HOLDOUT_WINDOW_MONTHS)
    if len(in_sample_idx) == 0:
        raise ValueError("in-sample window is empty after the holdout split")

    in_sample_assets = {k: f.loc[f.index.isin(in_sample_idx)] for k, f in aligned_assets.items()}
    in_sample_signals = {k: f.loc[f.index.isin(in_sample_idx)] for k, f in aligned_signals.items()}

    target = mod.make_target_fn({**in_sample_assets, **in_sample_signals})
    trades, active = rotation_engine.run_two_asset_rotation(
        in_sample_assets, target,
        cost_bps_by_asset={mod.TRADED[0]: cost_bps, mod.TRADED[1]: cost_bps},
        signal_frames=in_sample_signals)

    n_trades = len(trades)
    result = {
        "in_sample_sessions": len(in_sample_idx),
        "holdout_sessions_excluded": len(holdout_idx),
        "in_sample_start": str(in_sample_idx.min().date()) if len(in_sample_idx) else None,
        "in_sample_end": str(in_sample_idx.max().date()) if len(in_sample_idx) else None,
        "n_trades": n_trades,
        "trades_by_asset": trades["asset"].value_counts().to_dict() if n_trades else {},
        "gross_ret_mean": float(trades["gross_ret"].mean()) if n_trades else None,
        "net_ret_mean": float(trades["net_ret"].mean()) if n_trades else None,
        "net_ret_sum": float(trades["net_ret"].sum()) if n_trades else None,
        "sessions_in_market": int(active.notna().sum()),
        "sessions_total": len(active),
    }
    return result


def prepare_score_call_description() -> dict:
    """Describes, without importing or calling any of it, the ABI-1
    scoring interface (curve builder, evaluator, ledger writer) for this
    family's rotation shape."""
    return {
        "module": "mining.abi1_driver",
        "curve_builder": "build_rotation_curve(asset_frames, target_fn, "
                         "cost_bps_by_asset=..., signal_frames=...)",
        "evaluator": "evaluate_family(curve, benchmark_frame=<SPY full "
                     "history>, benchmark_entry_cost_bps=..., "
                     "account_type=<account type>, "
                     "tax_policy='realization')",
        "ledger_writer": "record_abi1_outcome(trial_ledger path, "
                         f"family_id={mod.FAMILY_ID!r}, ...)",
        "status": "description only -- this script does not import or "
                  "call these functions (see its module docstring)",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--xlk-bid", type=float, required=True)
    parser.add_argument("--xlk-ask", type=float, required=True)
    parser.add_argument("--prepare-score-call", action="store_true")
    args = parser.parse_args(argv)

    cost_bps = cost_tax_inputs.session_round_trip(
        cost_tax_inputs.compute_spread_bps(args.xlk_bid, args.xlk_ask),
        cost_tax_inputs.compute_spread_bps(args.xlk_bid, args.xlk_ask))

    print(f"[step2] XLK quote bid={args.xlk_bid} ask={args.xlk_ask} -> "
          f"round-trip cost_bps={cost_bps:.4f} (applied to both legs)")

    irx = data_pipeline.fetch_risk_free_rate(start="max")
    asset_frames = fetch_traded_legs(irx_series=irx)
    signal_frames = fetch_signal_legs()

    try:
        aligned_assets, aligned_signals = align_to_common_calendar(
            asset_frames, signal_frames)
    except ValueError as exc:
        print(f"[step3] BLOCKED -- {exc}")
        payload = {"step2_cost_bps": cost_bps, "step3": "blocked",
                   "step3_reason": str(exc)}
        if args.prepare_score_call:
            payload["step4_prepared_not_executed"] = prepare_score_call_description()
        print(json.dumps(payload, indent=2))
        return 1

    result = run_rough_backtest(aligned_assets, aligned_signals, cost_bps=cost_bps)
    payload = {"step2_cost_bps": cost_bps, "step3": result}
    if args.prepare_score_call:
        payload["step4_prepared_not_executed"] = prepare_score_call_description()
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
