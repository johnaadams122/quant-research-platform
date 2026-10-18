"""Catalog-driven, ledger-safe batch3 mining runner.

The feasibility matrix freezes what may be tested.  The catalog freezes each
family's economic contract.  This driver joins the two, loads every required
input before invoking a backtest, computes the whole batch, and only then
applies one order-independent multiple-testing correction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Callable, Mapping
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import data_pipeline
import trading_calendar
from mining import (backtest_engine, catalog, constants, cost_tax_inputs, dsr,
                    evidence, external_data, near_miss, results_log,
                    rotation_engine, ruled_out, signals, trial_ledger)


LEDGER_PATH = os.path.join(REPO_ROOT, "data", "mining", "trial_ledger.jsonl")
CATALOG_PATH = os.path.join(REPO_ROOT, "data", "mining", "catalog.json")
RESEARCH_ROOT = os.path.join(REPO_ROOT, "runs", "mining")
RESULTS_LOG_PATH = os.path.join(RESEARCH_ROOT, "RESULTS-LOG.md")
NEAR_MISS_LOG_PATH = os.path.join(RESEARCH_ROOT, "NEAR-MISS-LOG.md")
RUN_DIR = os.path.join(RESEARCH_ROOT, "batch3")
MATRIX_PATH = os.path.join(RUN_DIR, "feasibility-matrix.json")
DRY_RUN_PATH = os.path.join(RUN_DIR, "batch3-dry-run.json")
REAL_RUN_PATH = os.path.join(RUN_DIR, "batch3-results.json")
RAW_DATA_DIR = os.path.join(REPO_ROOT, "data", "raw")
EXPECTED_MATRIX_CANONICAL_SHA256 = (
    "ef8573a2c8c32769cc3dd82d1040fea8e152e66dc01742e01f3c68ef14bde51a")
# The evidence lifecycle's `signal_source_path`/`signal_source_sha256` drift
# check needs ONE file to hash per candidate. This
# script's own dispatch logic (_single_factory/_rotation_target) decides
# WHICH signals.py function backs a given family_id, so a change to that
# dispatch is itself a real change to what gets tested even when signals.py
# is untouched -- hashing this file, not signals.py alone, is the more
# conservative choice. Deliberate simplification, disclosed rather than
# hidden: _score_one is also imported unmodified by run_swing_mining.py and
# run_day_trading_mining.py, whose real signal sources
# live in different files -- this constant is the same "the module
# implementing the score-decision logic" answer for all three, not a
# per-family-precise one.
_SIGNAL_SOURCE_PATH = os.path.relpath(__file__, REPO_ROOT).replace(os.sep, "/")

# Calendar-day buffer prepended to the ROTATION holdout bucket so a
# target_fn with real trailing-lookback needs (e.g. a 252-session/12-month
# moving-average or absolute-momentum window) sees genuine history instead
# of silently starving for the whole holdout period. Without this, the bare
# holdout_start:holdout_end slice is at most ~130 trading days (the
# pipeline's own HOLDOUT_WINDOW_MONTHS=6), which can never satisfy a
# 252-session lookback -- the signal returns None on every holdout decision,
# producing zero holdout trades that then read as a real "no edge" failure
# instead of "never actually exercised." Same mechanism as
# scripts/run_expanded_universe_mining.py's HOLDOUT_LOOKBACK_BUFFER_DAYS.
#
# Scope note: _score_one has TWO early-return branches that both land on
# stage="cut_dsr" before ever reaching the `elif backtest_engine.
# holdout_passes(...)` branch a starved lookback could corrupt: (1) `if stats[
# "trade_count"] < constants.MIN_TRADES_FOR_SCORING` (in-sample trade count
# only, dsr_value left None) and (2) `elif value < constants.
# DSR_PASS_THRESHOLD` (a DSR computed purely from in-sample data). Verdicts
# from those two branches never depend on holdout_net_edge_bps. This buffer
# matters for a candidate that clears the DSR gate and would otherwise
# get a false "no holdout edge" cut.
#
# 600 days (not 400): 400 calendar days can leave a 252-session lookback
# family's earliest monthly holdout decisions short of their trailing
# window, or clearing it by only a handful of sessions -- "with margin" is
# not true at 400. A family's real requirement is
# (its own lookback parameter in trading sessions) + up to ~21 sessions lost
# to monthly-anchor quantization (a decision frozen at a month's first
# session effectively "wastes" the rest of that month) + real NYSE holidays.
# 600 calendar days gives roughly 410-420 real trading sessions of padding
# (600 * ~0.69, the typical trading-day fraction),
# comfortably covering every catalog family's stated lookback (max 252
# sessions) plus quantization loss with real margin, not a razor edge.
#
# A trade opened during the padding period is handled two ways, not just
# filtered: one with zero overlap with the real holdout window is a pure
# warm-up artifact and is dropped; one held INTO the real holdout window is
# marked to market at the first real holdout session's open (see
# _mark_to_market_spanning_trades) so a low-turnover/trending signal that
# doesn't switch positions during holdout isn't misread as inactive
# (dropping every warm-up-originated trade outright would misread it).
HOLDOUT_LOOKBACK_BUFFER_DAYS = 600


EXPECTED_SCOPE_IDS = (
    "analyst_rating_upgrade_momentum",
    "insider_cluster_buying_value_overlay",
    "hy_credit_bond_absolute_momentum",
    "treasury_duration_volatility_timing",
    "tips_nominal_breakeven_rotation",
    "intl_govt_bond_time_series_momentum",
    "stock_split_signaling_drift",
    "despac_post_merger_fade",
    "activist_13d_campaign_drift",
    "free_cash_flow_yield_cows",
    "wide_moat_quality_investing",
    "faith_based_sin_stock_screen",
    "human_capital_employee_satisfaction",
    "skew_index_tail_risk_regime",
    "monthly_opex_week_drift",
    "vxn_vix_spread_derisking_regime",
    "margin_debt_growth_regime",
    "aaii_cash_allocation_contrarian",
    "money_market_fund_flow_riskoff",
    "fed_net_liquidity_regime",
)


def _contract(mechanism_class, catalog_tickers, traded, signal,
              mechanism_sha256, parameter_types):
    return {
        "mechanism_class": mechanism_class,
        "catalog_tickers": tuple(catalog_tickers),
        "required_traded_symbols": tuple(traded),
        "signal_symbols": tuple(signal),
        "mechanism_sha256": mechanism_sha256,
        "parameter_types": parameter_types,
        "account_types": ("taxable", "roth"),
    }


SUPPORTED_CONTRACTS = {
    "hy_credit_bond_absolute_momentum": _contract(
        "monthly_two_asset_absolute_momentum_rotation", ("HYG", "BIL"),
        ("HYG", "BIL"), ("HYG total return",),
        "c42ee463af85db9610348b4c2c1cb13b6d1f0fd96504458751bac843291ae246",
        {"lookback_months": int, "signal": str, "rebalance": str,
         "cash_proxy": str}),
    "treasury_duration_volatility_timing": _contract(
        "monthly_two_asset_realized_volatility_rotation", ("TLT", "SHY"),
        ("TLT", "SHY"), ("TLT realized volatility",),
        "afe706354cf83ba53db85a5f964e9234b6add789542d7d2ebd23bd2e2a9c648f",
        {"vol_lookback_days": int, "vol_threshold": str,
         "rebalance": str}),
    "tips_nominal_breakeven_rotation": _contract(
        "monthly_two_asset_relative_momentum_rotation", ("TIP", "IEF"),
        ("TIP", "IEF"), ("TIP minus IEF trailing total return",),
        "64f5278d89df39a5427c7e85531bce8b6fce9becd31f49f92b61af5eee7f5410",
        {"lookback_months": int, "signal": str, "rebalance": str}),
    "free_cash_flow_yield_cows": _contract(
        "single_primary_proxy_buy_and_hold", ("COWZ", "DSTL"), ("COWZ",), (),
        "4a016270f2e7a5b752d911daabdda10ba2b5543c22319db4d58f423facd47bd6",
        {"screen_metric": str, "portfolio_size": int, "rebalance": str}),
    "wide_moat_quality_investing": _contract(
        "single_proxy_buy_and_hold", ("MOAT",), ("MOAT",), (),
        "1ee48e3514399f82f01948dafe61295301b3c83fafc68e088495c6673debb650",
        {"screen_metric": str, "moat_rating_required": str,
         "valuation_filter": str, "rebalance": str}),
    "skew_index_tail_risk_regime": _contract(
        "daily_signal_two_asset_regime_rotation", ("SPY", "^SKEW"),
        ("SPY", "SHY"), ("^SKEW",),
        "bebc67a20c7304b83809704eb4ca46a4d30ec5fd7dfd2d0fcb55dc9eb23693d5",
        {"signal_series": str, "skew_fast_ma_days": int,
         "skew_slow_ma_days": int, "defensive_proxy": str}),
    "monthly_opex_week_drift": _contract(
        "calendar_window_long_cash", ("SPY",), ("SPY",),
        ("third-Friday exchange calendar",),
        "822a4dc5e07a9fc79e357a77d9a5af2116120310edfcf86945be70fcdb20a9b9",
        {"opex_week_definition": str,
         "entry_trading_days_before_opex_friday": int, "exit_at": str}),
    "vxn_vix_spread_derisking_regime": _contract(
        "daily_ratio_signal_two_asset_regime_rotation",
        ("QQQ", "^VXN", "^VIX"), ("QQQ", "SHY"), ("^VXN", "^VIX"),
        "7a239c724771513efcb8908ceddd263fae82fd87308291f22e5f4cefde16b2f8",
        {"signal_ratio": str, "ratio_fast_ma_days": int,
         "ratio_slow_ma_days": int, "defensive_proxy": str}),
    "margin_debt_growth_regime": _contract(
        "monthly_external_signal_long_cash", ("SPY",), ("SPY",),
        ("FINRA aggregate customer margin debt",),
        "1a69dc6841d9c762beabf973c6e3195e2fe44a9827bcef98555633a56ce6d83e",
        {"margin_debt_lookback_months": int,
         "signal_lag_trading_days": int,
         "growth_threshold": (int, float)}),
}

EXPECTED_PARAMETER_VALUES = {
    "hy_credit_bond_absolute_momentum": {
        "lookback_months": 6,
        "signal": "trailing_total_return",
        "rebalance": "monthly",
        "cash_proxy": "BIL",
    },
    "treasury_duration_volatility_timing": {
        "vol_lookback_days": 20,
        "vol_threshold": "trailing_252d_median",
        "rebalance": "monthly",
    },
    "tips_nominal_breakeven_rotation": {
        "lookback_months": 3,
        "signal": "TIP_minus_IEF_trailing_return",
        "rebalance": "monthly",
    },
    "free_cash_flow_yield_cows": {
        "screen_metric": "free_cash_flow_yield",
        "portfolio_size": 100,
        "rebalance": "quarterly",
    },
    "wide_moat_quality_investing": {
        "screen_metric": "morningstar_economic_moat_rating",
        "moat_rating_required": "wide",
        "valuation_filter": "price_to_fair_value",
        "rebalance": "semiannual_staggered",
    },
    "skew_index_tail_risk_regime": {
        "signal_series": "SKEW",
        "skew_fast_ma_days": 20,
        "skew_slow_ma_days": 252,
        "defensive_proxy": "SHY",
    },
    "monthly_opex_week_drift": {
        "opex_week_definition": "week_containing_third_friday_of_month",
        "entry_trading_days_before_opex_friday": 4,
        "exit_at": "opex_friday_close",
    },
    "vxn_vix_spread_derisking_regime": {
        "signal_ratio": "VXN/VIX",
        "ratio_fast_ma_days": 20,
        "ratio_slow_ma_days": 252,
        "defensive_proxy": "SHY",
    },
    "margin_debt_growth_regime": {
        "margin_debt_lookback_months": 12,
        "signal_lag_trading_days": 20,
        "growth_threshold": 0,
    },
}


def _is_expected_type(value, expected) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, expected)


def validate_supported_contract(entry: dict, matrix_row: dict) -> None:
    """Fail closed if the reviewed execution contract drifts."""
    family_id = entry.get("family_id")
    if family_id not in SUPPORTED_CONTRACTS:
        raise ValueError(f"{family_id}: no reviewed supported contract")
    expected = SUPPORTED_CONTRACTS[family_id]
    comparisons = {
        "tickers": (tuple(entry.get("tickers", ())), expected["catalog_tickers"]),
        "account_types": (tuple(entry.get("account_types", ())),
                          expected["account_types"]),
        "mechanism_class": (matrix_row.get("mechanism_class"),
                            expected["mechanism_class"]),
        "catalog_symbols": (tuple(matrix_row.get("catalog_symbols", ())),
                            expected["catalog_tickers"]),
        "required_traded_symbols": (
            tuple(matrix_row.get("required_traded_symbols", ())),
            expected["required_traded_symbols"]),
        "signal_symbols": (tuple(matrix_row.get("signal_symbols", ())),
                           expected["signal_symbols"]),
    }
    for name, (observed, wanted) in comparisons.items():
        if observed != wanted:
            raise ValueError(
                f"{family_id}: {name} contract drift: {observed!r} != {wanted!r}")
    digest = hashlib.sha256(entry.get("mechanism", "").encode("utf-8")).hexdigest()
    if digest != expected["mechanism_sha256"]:
        raise ValueError(f"{family_id}: mechanism contract drift")
    params = entry.get("parameters")
    if not isinstance(params, dict):
        raise ValueError(f"{family_id}: parameters must be an object")
    schema = expected["parameter_types"]
    if set(params) != set(schema):
        raise ValueError(
            f"{family_id}: parameter keys drift: {sorted(params)} != "
            f"{sorted(schema)}")
    for name, expected_type in schema.items():
        if not _is_expected_type(params[name], expected_type):
            raise ValueError(
                f"{family_id}: {name} must be {expected_type}, got "
                f"{type(params[name]).__name__}")
    expected_values = EXPECTED_PARAMETER_VALUES[family_id]
    for name, expected_value in expected_values.items():
        if params[name] != expected_value:
            raise ValueError(
                f"{family_id}: {name} semantic contract drift: "
                f"{params[name]!r} != {expected_value!r}")


def catalog_parameters(candidate: dict) -> dict:
    """Translate only catalog-owned values into reviewed factory arguments."""
    family_id = candidate["family_id"]
    p = candidate["catalog"]["parameters"]
    if family_id in ("hy_credit_bond_absolute_momentum",
                     "tips_nominal_breakeven_rotation"):
        return {"lookback_months": p["lookback_months"]}
    if family_id == "treasury_duration_volatility_timing":
        match = re.fullmatch(r"trailing_(\d+)d_median", p["vol_threshold"])
        if match is None:
            raise ValueError(f"{family_id}: unreviewed vol_threshold format")
        return {"vol_lookback_days": p["vol_lookback_days"],
                "median_lookback_days": int(match.group(1))}
    if family_id == "monthly_opex_week_drift":
        return {"entry_trading_days_before_opex":
                p["entry_trading_days_before_opex_friday"]}
    if family_id == "skew_index_tail_risk_regime":
        return {"fast": p["skew_fast_ma_days"],
                "slow": p["skew_slow_ma_days"],
                "defensive_ticker": p["defensive_proxy"]}
    if family_id == "vxn_vix_spread_derisking_regime":
        return {"fast": p["ratio_fast_ma_days"],
                "slow": p["ratio_slow_ma_days"],
                "defensive_ticker": p["defensive_proxy"]}
    if family_id == "margin_debt_growth_regime":
        return {"lookback_months": p["margin_debt_lookback_months"],
                "lag_sessions": p["signal_lag_trading_days"],
                "growth_threshold": float(p["growth_threshold"])}
    if family_id in ("free_cash_flow_yield_cows",
                     "wide_moat_quality_investing"):
        return {}
    raise ValueError(f"{family_id}: no catalog parameter adapter")


def derive_holdout_window(tested_date: str) -> dict:
    tested = pd.Timestamp(tested_date).tz_localize(None).normalize()
    start = tested - pd.DateOffset(months=constants.HOLDOUT_WINDOW_MONTHS)
    sessions = trading_calendar.load_default_sessions()
    tested_day = tested.date()
    if tested_day < sessions[0] or tested_day > sessions[-1]:
        raise ValueError(
            f"date {tested_day} outside vendored NYSE calendar coverage "
            f"{sessions[0]} through {sessions[-1]}")
    required = max(session for session in sessions if session <= tested_day)
    return {"holdout_start": start.strftime("%Y-%m-%d"),
            "holdout_end": tested.strftime("%Y-%m-%d"),
            "required_session": required.isoformat()}


def validate_feasibility_matrix(matrix: dict) -> None:
    gate = matrix.get("run_scope", {}).get("quote_gate_bps")
    if gate != 300.0:
        raise ValueError(
            f"feasibility matrix quote_gate_bps drift: {gate!r} != 300.0")
    canonical = json.dumps(
        matrix, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    if digest != EXPECTED_MATRIX_CANONICAL_SHA256:
        raise ValueError(
            "feasibility matrix semantic drift: canonical SHA-256 "
            f"{digest} != {EXPECTED_MATRIX_CANONICAL_SHA256}")


def load_run_scope(catalog_path: str, matrix_path: str,
                   ledger_path: str) -> dict:
    entries = catalog.load_catalog(catalog_path)
    eligible, excluded = ruled_out.filter_eligible(entries, ledger_path)
    with open(matrix_path, "r", encoding="utf-8") as handle:
        matrix = json.load(handle)
    validate_feasibility_matrix(matrix)
    rows = matrix.get("candidates")
    if not isinstance(rows, list):
        raise ValueError("feasibility matrix candidates must be a list")
    ids = [row.get("family_id") for row in rows]
    if tuple(ids) != EXPECTED_SCOPE_IDS or len(set(ids)) != len(ids):
        raise ValueError("feasibility matrix must preserve the exact 20-family scope")
    if matrix.get("run_scope", {}).get("catalog_count") != len(ids):
        raise ValueError("feasibility matrix catalog_count disagrees with rows")
    by_id = {entry["family_id"]: entry for entry in entries}
    missing = [family_id for family_id in ids if family_id not in by_id]
    if missing:
        raise ValueError(f"matrix families missing from catalog: {missing}")
    for row in rows:
        family_id = row["family_id"]
        if row.get("provisional_decision") == "skip":
            continue
        if not trial_ledger.has_been_tested(ledger_path, family_id):
            continue
        # A not_evaluated_error family is a caught DEFECT, not evidence
        # about the market -- it stays eligible to re-enter batch3's
        # scope once the defect is fixed, matching ruled_out.
        # filter_eligible's own treatment at the catalog-filter level.
        # score_batch's own already_pinned check (downstream of this
        # function) is the authoritative safety net that verifies the
        # retry's window actually matches what is pinned before scoring
        # proceeds -- this scope-level check only decides who is
        # CONSIDERED, not whether it is safe to score.
        if trial_ledger.latest_outcome(ledger_path, family_id) != "not_evaluated_error":
            raise ValueError(
                f"{family_id}: already pinned/tested; batch3 is first-test-only")
    eligible_ids = {entry["family_id"] for entry in eligible}
    excluded_by_id = {item["family_id"]: item["reason"] for item in excluded}
    pairs = []
    provisional, matrix_skips, eligibility_skips = [], [], []
    for row in rows:
        family_id = row["family_id"]
        entry = by_id[family_id]
        if tuple(row.get("catalog_symbols", ())) != tuple(entry["tickers"]):
            raise ValueError(f"{family_id}: matrix catalog_symbols drift")
        pair = {"family_id": family_id, "catalog": entry, "matrix": row}
        pairs.append(pair)
        if row.get("provisional_decision") == "skip":
            matrix_skips.append(pair)
        elif family_id not in eligible_ids:
            pair["eligibility_reason"] = excluded_by_id.get(
                family_id, "catalog/ledger gate excluded family")
            eligibility_skips.append(pair)
        else:
            validate_supported_contract(entry, row)
            provisional.append(pair)
    if len(provisional) + len(matrix_skips) + len(eligibility_skips) != 20:
        raise AssertionError("every scoped family must reconcile once")
    gate = float(matrix.get("run_scope", {}).get("quote_gate_bps", math.nan))
    if not math.isfinite(gate) or gate <= 0:
        raise ValueError("quote_gate_bps must be positive and finite")
    for pair in provisional:
        pair["execution_costs"] = selected_execution_costs(pair["matrix"], gate)
    return {"rows": rows, "pairs": pairs, "provisional": provisional,
            "matrix_skips": matrix_skips,
            "eligibility_skips": eligibility_skips,
            "gate_exclusions": excluded, "quote_gate_bps": gate,
            "matrix": matrix}


def selected_execution_costs(matrix_row: dict, quote_gate_bps: float) -> dict:
    """Select only top-level matrix quotes for actual traded legs."""
    required = tuple(matrix_row.get("required_traded_symbols", ()))
    quotes = matrix_row.get("quotes", [])
    by_symbol = {quote.get("symbol"): quote for quote in quotes}
    selected = {}
    for symbol in required:
        quote = by_symbol.get(symbol)
        if quote is None:
            raise ValueError(f"{matrix_row['family_id']}: no quote for {symbol}")
        try:
            bid, ask = float(quote["bid"]), float(quote["ask"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{matrix_row['family_id']}: {symbol} quote missing numeric bid/ask") from exc
        if not all(math.isfinite(value) and value > 0 for value in (bid, ask)):
            raise ValueError(f"{matrix_row['family_id']}: {symbol} quote must be positive")
        if ask <= bid:
            raise ValueError(f"{matrix_row['family_id']}: {symbol} quote is not ordered")
        spread = cost_tax_inputs.compute_spread_bps(bid, ask)
        if spread >= quote_gate_bps:
            raise ValueError(
                f"{matrix_row['family_id']}: {symbol} spread {spread:.6f} "
                f"is not below {quote_gate_bps:.6f} bps")
        captured = quote.get("spread_bps")
        if captured is None or not math.isclose(
                float(captured), spread, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError(
                f"{matrix_row['family_id']}: {symbol} captured spread disagrees "
                "with raw bid/ask")
        selected[symbol] = {
            "bid": bid, "ask": ask, "cost_bps": spread,
            "source": quote.get("source") or quote.get("preferred_source"),
        }
    if set(selected) != set(required):
        raise AssertionError("each traded leg must receive exactly one cost")
    return selected


def _normalize_ohlc(raw: pd.DataFrame, symbol: str, *,
                     require_positive: bool = True) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError(f"{symbol}: OHLC input is empty")
    frame = raw.copy()
    if "Date" in frame.columns:
        index = pd.DatetimeIndex(pd.to_datetime(frame.pop("Date")))
    else:
        index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    frame.index = index.normalize()
    required = ["Open", "High", "Low", "Close"]
    if any(column not in frame.columns for column in required):
        raise ValueError(f"{symbol}: OHLC columns must be {required}")
    frame = frame.loc[:, required]
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{symbol}: calendar must be sorted and unique")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        if require_positive:
            # A require_positive=True call site gets the one combined
            # "finite and positive" message for both failures.
            raise ValueError(f"{symbol}: OHLC values must be finite and positive")
        raise ValueError(f"{symbol}: OHLC values must be finite")
    if require_positive and (values <= 0).any():
        raise ValueError(f"{symbol}: OHLC values must be finite and positive")
    return frame


def _load_ohlc(symbol: str) -> pd.DataFrame:
    raw = data_pipeline.fetch_ohlc(symbol, start="max")
    if raw is None or raw.empty:
        cache = os.path.join(RAW_DATA_DIR, f"{symbol.lstrip('^')}_ohlc.csv")
        if not os.path.exists(cache):
            raise RuntimeError(f"no live or existing cached OHLC for {symbol}")
        raw = pd.read_csv(cache)
    return _normalize_ohlc(raw, symbol)


def _trade_stats(trades: pd.DataFrame) -> dict:
    count = int(len(trades))
    if count == 0:
        return {"trade_count": 0, "net_edge_bps": 0.0, "sharpe": 0.0,
                "skew": 0.0, "kurtosis": 3.0}
    returns = pd.to_numeric(trades["net_ret"], errors="raise").to_numpy(float)
    if not np.isfinite(returns).all():
        raise ValueError("trade returns must be finite")
    edge = float(np.mean(returns) * 1e4)
    std = float(np.std(returns, ddof=1)) if count >= 2 else 0.0
    if count < 2 or std <= 0 or not math.isfinite(std):
        return {"trade_count": count, "net_edge_bps": edge, "sharpe": 0.0,
                "skew": 0.0, "kurtosis": 3.0}
    skew = float(scipy.stats.skew(returns, bias=True))
    kurtosis = float(scipy.stats.kurtosis(
        returns, fisher=False, bias=True))
    if not math.isfinite(skew) or not math.isfinite(kurtosis):
        skew, kurtosis = 0.0, 3.0
    kurtosis = max(kurtosis, 1.0 + skew ** 2)
    return {"trade_count": count, "net_edge_bps": edge,
            "sharpe": float(np.mean(returns) / std), "skew": skew,
            "kurtosis": kurtosis}


def _window_masks(index, holdout_start, holdout_end):
    start, end = pd.Timestamp(holdout_start), pd.Timestamp(holdout_end)
    return index < start, (index >= start) & (index <= end)


def _require_bucket(frame: pd.DataFrame, name: str) -> None:
    if len(frame) < 2:
        raise ValueError(f"{name} has fewer than two bars")


def run_single_asset_windows(frame: pd.DataFrame, signal_factory: Callable, *,
                             holdout_start: str, holdout_end: str,
                             cost_bps: float,
                             tax_haircut_rate: float) -> dict:
    frame = _normalize_ohlc(frame, "single_asset")
    is_mask, hold_mask = _window_masks(
        frame.index, holdout_start, holdout_end)
    buckets = {"in_sample": frame.loc[is_mask].copy(),
               "holdout": frame.loc[hold_mask].copy()}
    outputs = {}
    for name, bucket in buckets.items():
        _require_bucket(bucket, name)
        signal_fn = signal_factory(bucket.index.copy())
        trades, _ = backtest_engine.run_backtest(
            bucket, signal_fn, cost_bps=cost_bps,
            tax_haircut_rate=tax_haircut_rate)
        outputs[f"{name}_trades"] = trades
        outputs[f"{name}_index"] = bucket.index
        outputs[f"stats_{name}"] = _trade_stats(trades)
    outputs["holdout_net_edge_bps"] = outputs["stats_holdout"]["net_edge_bps"]
    return outputs


def _rotation_bucket(asset_frames: Mapping[str, pd.DataFrame],
                     signal_frames: Mapping[str, pd.DataFrame], mask_fn,
                     bucket_name: str):
    assets = {key: frame.loc[mask_fn(frame.index)].copy()
              for key, frame in asset_frames.items()}
    for key, frame in assets.items():
        _require_bucket(frame, f"{bucket_name} {key}")
    calendars = [frame.index for frame in assets.values()]
    if any(not index.equals(calendars[0]) for index in calendars[1:]):
        raise ValueError(
            f"{bucket_name} traded legs must share an identical calendar")
    original_calendar = calendars[0]
    signals_in_bucket = {
        key: frame.loc[mask_fn(frame.index)].copy()
        for key, frame in signal_frames.items()
    }
    common = original_calendar
    for frame in signals_in_bucket.values():
        common = common.intersection(frame.index, sort=False)
    if len(common) < 2:
        raise ValueError(f"{bucket_name} aligned signal calendar has <2 bars")
    dropped = len(original_calendar) - len(common)
    assets = {key: frame.loc[common].copy() for key, frame in assets.items()}
    aligned_signals = {key: frame.loc[common].copy()
                       for key, frame in signals_in_bucket.items()}
    return assets, aligned_signals, {
        "traded_dates_before_signal_alignment": len(original_calendar),
        "aligned_dates": len(common), "signal_dates_dropped": dropped,
    }


def _normalize_rotation_inception(
        asset_frames: Mapping[str, pd.DataFrame],
        holdout_start: pd.Timestamp):
    raw_starts = {key: frame.index.min() for key, frame in asset_frames.items()}
    common_inception = max(raw_starts.values())
    if common_inception >= holdout_start:
        raise ValueError(
            f"rotation common inception {common_inception.date()} must be "
            f"strictly before holdout start {holdout_start.date()}")
    trimmed_counts = {
        key: int((frame.index < common_inception).sum())
        for key, frame in asset_frames.items()}
    normalized = {
        key: frame.loc[frame.index >= common_inception].copy()
        for key, frame in asset_frames.items()}
    evidence = {
        "raw_leg_starts": {
            key: value.strftime("%Y-%m-%d") for key, value in raw_starts.items()},
        "common_inception": common_inception.strftime("%Y-%m-%d"),
        "sessions_trimmed_by_leg": trimmed_counts,
    }
    return normalized, evidence


def _mark_to_market_spanning_trades(
        trades: pd.DataFrame, bucket_assets: Mapping[str, pd.DataFrame],
        start: pd.Timestamp, cost_bps_by_asset: Mapping[str, float],
        tax_haircut_rate: float) -> tuple[pd.DataFrame, int, int]:
    """Reconcile trades computed on a lookback-padded holdout bucket against
    the REAL holdout window starting at ``start``.

    A trade entered before ``start`` and also closed before ``start`` is a
    pure warm-up artifact with zero overlap with real holdout and is
    dropped. A trade entered before ``start`` but held INTO the real holdout
    window is neither dropped (which understates real holdout activity for
    any low-turnover/trending signal that doesn't switch positions during
    holdout) nor kept with its true pre-holdout entry
    (which would leak padding-period return into a holdout metric). It is
    marked to market instead: re-priced as if freshly opened at the first
    REAL holdout session's open -- the same "enter at Open" convention
    rotation_engine.py itself uses -- keeping the position's true real exit
    and exactly the ONE real round-trip cost paid at that exit (no new cost
    is invented for the synthetic re-entry, since no real transaction
    happens at a mark-to-market boundary).

    Returns (real_trades, pure_warmup_dropped_count, marked_to_market_count).
    """
    columns = list(trades.columns)
    if len(trades) == 0:
        return trades, 0, 0
    entry = pd.to_datetime(trades["entry_date"])
    exit_ = pd.to_datetime(trades["exit_date"])
    real = trades.loc[entry >= start].copy()
    # A trade whose exit lands EXACTLY on `start` closed at that session's
    # open (rotation_engine's own "switch" pricing convention) and so had
    # zero real holdout exposure -- it must count as pure warm-up (dropped),
    # not spanning, or marking it to market would synthesize an entry and
    # exit at the identical price (gross_ret == 0.0) while still charging a
    # real round-trip cost for zero real activity, misreading "never
    # actually exercised" as "exercised and mildly cost-negative." (Hence
    # `exit_ > start` below, not `>=`.)
    spanning = trades.loc[(entry < start) & (exit_ > start)]
    pure_warmup_count = int(((entry < start) & (exit_ <= start)).sum())
    marked_rows = []
    for _, row in spanning.iterrows():
        asset = row["asset"]
        frame = bucket_assets[asset]
        real_holdout_rows = frame.loc[frame.index >= start]
        if real_holdout_rows.empty:
            continue
        synthetic_entry_date = real_holdout_rows.index[0]
        synthetic_entry_px = float(real_holdout_rows.iloc[0]["Open"])
        exit_px = float(row["exit_px"])
        gross_ret = exit_px / synthetic_entry_px - 1.0
        cost_bps = float(cost_bps_by_asset[asset])
        net_ret = gross_ret - cost_bps / 1e4
        if net_ret > 0.0 and tax_haircut_rate:
            net_ret *= 1.0 - tax_haircut_rate
        held = (frame.index.get_loc(row["exit_date"])
                - frame.index.get_loc(synthetic_entry_date))
        marked_rows.append({
            "asset": asset, "entry_date": synthetic_entry_date,
            "exit_date": row["exit_date"], "entry_px": synthetic_entry_px,
            "exit_px": exit_px, "held": held, "reason": row["reason"],
            "gross_ret": gross_ret, "net_ret": net_ret,
        })
    if marked_rows:
        marked_df = pd.DataFrame(marked_rows, columns=columns)
        marked_df["entry_date"] = pd.to_datetime(marked_df["entry_date"])
        marked_df["exit_date"] = pd.to_datetime(marked_df["exit_date"])
        real = pd.concat([real, marked_df], ignore_index=True)
    real = real.sort_values("entry_date").reset_index(drop=True)
    return real, pure_warmup_count, len(marked_rows)


def run_rotation_windows(asset_frames: Mapping[str, pd.DataFrame],
                         signal_frames: Mapping[str, pd.DataFrame],
                         target_fn: Callable, *, holdout_start: str,
                         holdout_end: str,
                         cost_bps_by_asset: Mapping[str, float],
                         tax_haircut_rate: float,
                         expected_holdout_calendar: pd.DatetimeIndex | None = None
                         ) -> dict:
    assets = {key: _normalize_ohlc(frame, key)
              for key, frame in asset_frames.items()}
    signal_inputs = {key: _normalize_ohlc(frame, key)
                     for key, frame in signal_frames.items()}
    start, end = pd.Timestamp(holdout_start), pd.Timestamp(holdout_end)
    assets, normalization = _normalize_rotation_inception(assets, start)
    if expected_holdout_calendar is not None:
        expected = pd.DatetimeIndex(
            pd.to_datetime(expected_holdout_calendar)).tz_localize(None)
        expected = expected[(expected >= start) & (expected <= end)]
        if expected.empty or not expected.is_monotonic_increasing or expected.has_duplicates:
            raise ValueError("expected holdout calendar must be sorted and nonempty")
        for key, frame in assets.items():
            observed = frame.index[(frame.index >= start) & (frame.index <= end)]
            if not observed.equals(expected):
                raise ValueError(
                    f"holdout {key} does not match expected holdout calendar")
    padded_start = start - pd.Timedelta(days=HOLDOUT_LOOKBACK_BUFFER_DAYS)
    maskers = {"in_sample": lambda index: index < start,
               "holdout": lambda index: (index >= padded_start) & (index <= end)}
    outputs, alignment = {}, {}
    for name, mask_fn in maskers.items():
        bucket_assets, bucket_signals, details = _rotation_bucket(
            assets, signal_inputs, mask_fn, name)
        trades, _ = rotation_engine.run_two_asset_rotation(
            bucket_assets, target_fn,
            cost_bps_by_asset=cost_bps_by_asset,
            tax_haircut_rate=tax_haircut_rate,
            signal_frames=bucket_signals)
        index = next(iter(bucket_assets.values())).index
        if name == "holdout":
            # Reconcile the lookback-padded computation back to the REAL
            # holdout window: drop pure warm-up trades, mark to market any
            # trade held into real holdout so a non-switching signal isn't
            # misread as inactive (see _mark_to_market_spanning_trades).
            trades, warmup_dropped, marked_to_market = (
                _mark_to_market_spanning_trades(
                    trades, bucket_assets, start, cost_bps_by_asset,
                    tax_haircut_rate))
            index = index[index >= start]
            details = {**details,
                       "lookback_buffer_days": HOLDOUT_LOOKBACK_BUFFER_DAYS,
                       "padded_start": padded_start.strftime("%Y-%m-%d"),
                       "warmup_trades_dropped": warmup_dropped,
                       "warmup_trades_marked_to_market": marked_to_market}
        outputs[f"{name}_trades"] = trades
        outputs[f"{name}_index"] = index
        outputs[f"stats_{name}"] = _trade_stats(trades)
        alignment[name] = details
    outputs["holdout_net_edge_bps"] = outputs["stats_holdout"]["net_edge_bps"]
    alignment["normalization"] = normalization
    outputs["alignment"] = alignment
    return outputs


def _expected_holdout_calendar(holdout_start: str,
                               holdout_end: str) -> pd.DatetimeIndex:
    start, end = pd.Timestamp(holdout_start).date(), pd.Timestamp(holdout_end).date()
    sessions = trading_calendar.load_default_sessions()
    expected = [session for session in sessions if start <= session <= end]
    if not expected:
        raise ValueError("fixed holdout has no covered NYSE sessions")
    return pd.DatetimeIndex(pd.to_datetime(expected))


def _always_long_factory(_index):
    def signal_fn(_frame, _i):
        return backtest_engine.SignalResult(entry=True, exit=False)
    return signal_fn


def _single_factory(candidate: dict, finra_state: pd.Series | None):
    family_id = candidate["family_id"]
    params = catalog_parameters(candidate)
    if family_id in ("free_cash_flow_yield_cows",
                     "wide_moat_quality_investing"):
        return _always_long_factory
    if family_id == "monthly_opex_week_drift":
        def factory(index):
            entry, exit_ = signals.monthly_opex_dates(index, **params)
            return signals.make_date_window_signal_fn(entry, exit_)
        return factory
    if family_id == "margin_debt_growth_regime":
        if not isinstance(finra_state, pd.Series) or finra_state.empty:
            raise ValueError("FINRA candidate requires a prepared causal state")
        def factory(index):
            released = finra_state.loc[
                (finra_state.index >= index.min()) &
                (finra_state.index <= index.max())].copy()
            signal_fn = signals.make_external_regime_signal_fn(released)
            if getattr(signal_fn, "ledgerable", True) is False:
                raise ValueError("external signal callable is non-ledgerable")
            return signal_fn
        return factory
    raise ValueError(f"{family_id}: not a single-asset candidate")


def _rotation_target(candidate: dict):
    family_id = candidate["family_id"]
    traded = candidate["matrix"]["required_traded_symbols"]
    params = catalog_parameters(candidate)
    risk, defensive = traded
    if params.get("defensive_ticker", defensive) != defensive:
        raise ValueError(f"{family_id}: catalog defensive proxy drift")
    if family_id == "hy_credit_bond_absolute_momentum":
        return signals.make_monthly_absolute_momentum_target_fn(
            risk, defensive, **params)
    if family_id == "tips_nominal_breakeven_rotation":
        return signals.make_monthly_relative_momentum_target_fn(
            risk, defensive, **params)
    if family_id == "treasury_duration_volatility_timing":
        return signals.make_monthly_volatility_target_fn(
            risk, defensive, **params)
    if family_id == "skew_index_tail_risk_regime":
        return signals.make_moving_average_regime_target_fn(
            risk_ticker=risk, defensive_ticker=defensive,
            signal_ticker=candidate["matrix"]["signal_symbols"][0],
            fast=params["fast"], slow=params["slow"])
    if family_id == "vxn_vix_spread_derisking_regime":
        numerator, denominator = candidate["matrix"]["signal_symbols"]
        return signals.make_ratio_regime_target_fn(
            risk_ticker=risk, defensive_ticker=defensive,
            numerator_ticker=numerator, denominator_ticker=denominator,
            fast=params["fast"], slow=params["slow"])
    raise ValueError(f"{family_id}: not a two-asset rotation")


def compute_candidate(candidate: dict, frames: dict,
                      finra_snapshot: external_data.ExternalSeriesSnapshot | None,
                      window: dict, *, finra_state: pd.Series | None = None) -> dict:
    family_id = candidate["family_id"]
    entry = candidate["catalog"]
    row = candidate["matrix"]
    validate_supported_contract(entry, row)
    costs = candidate.get("execution_costs") or selected_execution_costs(
        row, 300.0)
    tax_rate = cost_tax_inputs.tax_haircut_rate_for(entry["account_types"][0])
    traded = row["required_traded_symbols"]
    signal_symbols = [symbol for symbol in row["signal_symbols"] if symbol.startswith("^")]
    if len(traded) == 1:
        symbol = traded[0]
        if family_id == "margin_debt_growth_regime" and finra_state is None:
            finra_state = prepare_finra_state(
                finra_snapshot, candidate, frames[symbol],
                required_session=pd.Timestamp(window["required_session"]))
        windows = run_single_asset_windows(
            frames[symbol],
            _single_factory(candidate, finra_state),
            holdout_start=window["holdout_start"],
            holdout_end=window["holdout_end"],
            cost_bps=costs[symbol]["cost_bps"],
            tax_haircut_rate=tax_rate)
    else:
        windows = run_rotation_windows(
            {symbol: frames[symbol] for symbol in traded},
            {symbol: frames[symbol] for symbol in signal_symbols},
            _rotation_target(candidate),
            holdout_start=window["holdout_start"],
            holdout_end=window["holdout_end"],
            cost_bps_by_asset={symbol: costs[symbol]["cost_bps"]
                               for symbol in traded},
            tax_haircut_rate=tax_rate,
            expected_holdout_calendar=_expected_holdout_calendar(
                window["holdout_start"], window["holdout_end"]))
    daily = backtest_engine.daily_returns_by_exit_day(
        windows["in_sample_index"], windows["in_sample_trades"])
    daily = daily.tail(constants.DAILY_RETURNS_CORRELATION_WINDOW_DAYS)
    in_sample_index = windows["in_sample_index"]
    result = {
        "family_id": family_id, "ticker": "/".join(traded),
        # `tickers` is the REAL
        # traded instrument list, independent of the display `ticker`
        # string above -- evidence.build_insufficient_trades/
        # build_engine_error's instrument_set must be built from this, not
        # by re-parsing `ticker` (which a sibling producer,
        # run_swing_mining.py, can set to an arbitrary display label).
        "tickers": list(traded),
        "holdout_start": window["holdout_start"],
        "holdout_end": window["holdout_end"], "holdout_complete": True,
        "stats_in_sample": windows["stats_in_sample"],
        "stats_holdout": windows["stats_holdout"],
        "holdout_net_edge_bps": windows["holdout_net_edge_bps"],
        "daily_returns": daily,
        "account_type": entry["account_types"][0],
        "tax_haircut_rate": tax_rate,
        "execution_costs": costs,
        "data_through": {symbol: frames[symbol].index.max().strftime("%Y-%m-%d")
                         for symbol in traded + signal_symbols},
        "catalog_parameters": catalog_parameters(candidate),
        # `raw_catalog_parameters` is
        # the UNMAPPED catalog dict -- `catalog_parameters()` above is a
        # factory-ADAPTER view that renames/drops/casts keys per family
        # (e.g. monthly_opex_week_drift renames
        # entry_trading_days_before_opex_friday), so it misstates the
        # tested hypothesis if recorded as evidence. This is what
        # evidence.build_insufficient_trades/build_engine_error's
        # catalog_entry.parameters must read instead.
        "raw_catalog_parameters": dict(entry["parameters"]),
        "note": row["reason"],
        # Feeds evidence.build_insufficient_trades if this candidate is cut
        # for too few in-sample trades -- the SAME in-sample bucket _score_one's stats came from,
        # never recomputed, so the recorded window always matches what was
        # actually backtested.
        "in_sample_start": (in_sample_index.min().strftime("%Y-%m-%d")
                            if len(in_sample_index) else window["holdout_start"]),
        "in_sample_end": (in_sample_index.max().strftime("%Y-%m-%d")
                          if len(in_sample_index) else window["holdout_start"]),
    }
    if "alignment" in windows:
        result["signal_alignment"] = windows["alignment"]
    if family_id == "margin_debt_growth_regime":
        result["external_metadata"] = dict(finra_snapshot.metadata)
    return result


def _required_ohlc_symbols(provisional: list[dict]) -> list[str]:
    symbols = set()
    for candidate in provisional:
        symbols.update(candidate["matrix"]["required_traded_symbols"])
        symbols.update(symbol for symbol in candidate["matrix"]["signal_symbols"]
                       if symbol.startswith("^"))
    return sorted(symbols)


def validate_finra_coverage(
        snapshot: external_data.ExternalSeriesSnapshot, candidate: dict,
        traded_frame: pd.DataFrame, required_session: pd.Timestamp) -> dict:
    state = prepare_finra_state(
        snapshot, candidate, traded_frame,
        required_session=required_session)
    return dict(state.attrs["coverage_through_required_session"])


def validate_finra_snapshot(
        snapshot: external_data.ExternalSeriesSnapshot) -> None:
    if not isinstance(snapshot, external_data.ExternalSeriesSnapshot):
        raise ValueError("FINRA loader did not return ExternalSeriesSnapshot")
    metadata = snapshot.metadata
    expected = {
        "source_url": external_data.FINRA_MARGIN_STATISTICS_URL,
        "worksheet": external_data.FINRA_WORKSHEET,
        "units": "USD millions",
        "date_semantics": "reference_month_end",
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(
                f"FINRA {field} must be {value!r}, got {metadata.get(field)!r}")
    raw_sha256 = metadata.get("raw_sha256")
    if (not isinstance(raw_sha256, str) or
            re.fullmatch(r"[0-9a-fA-F]{64}", raw_sha256) is None):
        raise ValueError("FINRA raw_sha256 must be exactly 64 hexadecimal digits")
    retrieved_at = metadata.get("retrieved_at")
    try:
        parsed = datetime.fromisoformat(str(retrieved_at).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("FINRA retrieved_at must be a parseable timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("FINRA retrieved_at must include a timezone offset")
    if not isinstance(snapshot.series, pd.Series) or snapshot.series.empty:
        raise ValueError("FINRA series must be a nonempty pandas Series")


def prepare_finra_state(
        snapshot: external_data.ExternalSeriesSnapshot, candidate: dict,
        traded_frame: pd.DataFrame, *,
        required_session: pd.Timestamp | None = None) -> pd.Series:
    """Build one causal state from full history and the authoritative calendar."""
    validate_finra_snapshot(snapshot)
    params = catalog_parameters(candidate)
    balances = snapshot.series.copy()
    balances.attrs.update(snapshot.metadata)
    state = signals.finra_margin_debt_risk_on_series(
        balances, traded_frame.index, **params)
    if state.empty:
        raise ValueError("FINRA causal state is empty after catalog lookback/lag")
    if required_session is not None:
        required = pd.Timestamp(required_session).tz_localize(None).normalize()
        known = state.loc[state.index <= required]
        if known.empty:
            raise ValueError(
                "FINRA has no causal state on or before required session "
                f"{required.date()}")
        coverage = {
            "first_causal_state": known.index.min().strftime("%Y-%m-%d"),
            "latest_known_causal_state": known.index.max().strftime("%Y-%m-%d"),
            "required_session": required.strftime("%Y-%m-%d"),
        }
        state.attrs["coverage_through_required_session"] = coverage
        state.attrs["latest_known_on_or_before_required_session"] = (
            coverage["latest_known_causal_state"])
    return state


def compute_batch(scope: dict, *, tested_date: str,
                  ohlc_loader: Callable[[str], pd.DataFrame] = _load_ohlc,
                  finra_loader: Callable = external_data.fetch_finra_margin_debt) -> dict:
    """Load and preflight every input, then compute all feasible candidates."""
    window = derive_holdout_window(tested_date)
    required = pd.Timestamp(window["required_session"])
    frames, load_errors, coverage = {}, {}, {}
    for symbol in _required_ohlc_symbols(scope["provisional"]):
        try:
            frame = _normalize_ohlc(ohlc_loader(symbol), symbol)
            frames[symbol] = frame
            coverage[symbol] = {"first": frame.index.min().strftime("%Y-%m-%d"),
                                "last": frame.index.max().strftime("%Y-%m-%d")}
            if frame.index.max() < required:
                load_errors[symbol] = (
                    f"data ends {frame.index.max().date()}, before required "
                    f"session {required.date()}")
        except Exception as exc:
            load_errors[symbol] = f"{type(exc).__name__}: {exc}"

    candidate_preflight = {}
    for candidate in scope["provisional"]:
        symbols = list(candidate["matrix"]["required_traded_symbols"])
        symbols += [symbol for symbol in candidate["matrix"]["signal_symbols"]
                    if symbol.startswith("^")]
        problems = [f"{symbol}: {load_errors[symbol]}"
                    for symbol in symbols if symbol in load_errors]
        if problems:
            candidate_preflight[candidate["family_id"]] = "; ".join(problems)

    finra_snapshot, finra_state, finra_error = None, None, None
    finra_id = "margin_debt_growth_regime"
    finra_candidate = next((item for item in scope["provisional"]
                            if item["family_id"] == finra_id), None)
    if finra_candidate is not None and finra_id not in candidate_preflight:
        try:
            finra_snapshot = finra_loader()
            finra_state = prepare_finra_state(
                finra_snapshot, finra_candidate, frames["SPY"],
                required_session=required)
            coverage["FINRA_MARGIN_DEBT"] = dict(
                finra_state.attrs["coverage_through_required_session"])
        except Exception as exc:
            finra_error = f"{type(exc).__name__}: {exc}"
            candidate_preflight[finra_id] = finra_error

    reconciliation = []
    matrix_skip_ids = {pair["family_id"] for pair in scope["matrix_skips"]}
    eligibility = {pair["family_id"]: pair["eligibility_reason"]
                   for pair in scope["eligibility_skips"]}
    preflight_pairs = {pair["family_id"]: pair for pair in scope["provisional"]}
    computed = []
    for row in scope["rows"]:
        family_id = row["family_id"]
        if family_id in matrix_skip_ids:
            reconciliation.append({"family_id": family_id, "status": "skipped",
                                   "reason": row["reason"],
                                   "provisional_decision": "skip"})
        elif family_id in eligibility:
            reconciliation.append({"family_id": family_id, "status": "skipped",
                                   "reason": eligibility[family_id],
                                   "provisional_decision": row["provisional_decision"]})
        elif family_id in candidate_preflight:
            reconciliation.append({"family_id": family_id, "status": "skipped",
                                   "reason": candidate_preflight[family_id],
                                   "provisional_decision": row["provisional_decision"]})
        else:
            # This loop begins only after all OHLC and external inputs have been
            # loaded and checked.  No candidate can backtest during discovery.
            try:
                result = compute_candidate(
                    preflight_pairs[family_id], frames, finra_snapshot, window,
                    finra_state=(finra_state if family_id == finra_id else None))
                computed.append(result)
                reconciliation.append({"family_id": family_id,
                                       "status": "computed", "reason": None,
                                       "provisional_decision": row["provisional_decision"]})
            except Exception as exc:
                reconciliation.append({
                    "family_id": family_id, "status": "skipped",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "provisional_decision": row["provisional_decision"]})
    if len(reconciliation) != 20 or len({r["family_id"] for r in reconciliation}) != 20:
        raise AssertionError("20-family reconciliation is incomplete")
    return {"window": window, "computed": computed,
            "reconciliation": reconciliation, "input_coverage": coverage,
            "input_errors": load_errors,
            "external_metadata": (None if finra_snapshot is None else
                                  dict(finra_snapshot.metadata)),
            "finra_error": finra_error}


def compute_shared_correction(computed: list[dict], ledger_path: str) -> dict:
    ordered = sorted(computed, key=lambda item: item["family_id"])
    nondegenerate = []
    if ordered:
        union = pd.DatetimeIndex([])
        for item in ordered:
            series = item.get("daily_returns")
            if isinstance(series, pd.Series):
                union = union.union(pd.DatetimeIndex(series.index))
        union = union.sort_values()
        for item in ordered:
            series = item.get("daily_returns")
            if not isinstance(series, pd.Series) or series.empty:
                continue
            aligned = series.reindex(union, fill_value=0.0).to_numpy(dtype=float)
            if (aligned.size == 0 or not np.isfinite(aligned).all() or
                    np.std(aligned) == 0.0):
                continue
            nondegenerate.append(aligned)
    if nondegenerate:
        n_this = float(dsr.num_independent_trials(nondegenerate))
    else:
        n_this = float(len(ordered))
    # Delegated to trial_ledger.post_batch_floor: one source of truth for
    # this arithmetic, so a second copy elsewhere cannot drift from it.
    #
    # It reads the ledger ONCE, so the all-time count and the per-family
    # already-tested check come from the same snapshot: a family another
    # process first-tests mid-call cannot be counted by one read and missed
    # by the other.
    post_batch_floor = trial_ledger.post_batch_floor(
        ledger_path, [item["family_id"] for item in ordered])
    shared_n = float(max(n_this, post_batch_floor, 1))
    # sr_variance tiers (pooling is most principled, since
    # sr_variance is the DOMINANT term in the
    # DSR bar -- a straight multiplier, unlike n_trials's slowly-growing
    # factor). Preference order, each tier used only if the one above it
    # is unavailable: (1) pooled across every score this ledger has
    # ever recorded (trial_ledger.pooled_sr_variance) -- most data, least
    # sensitive to one run's arbitrary batch composition; (2) this run's
    # own candidate Sharpes (sample variance, needing at least two finite
    # values); (3) the pre-registered floor, last resort.
    pooled_variance = trial_ledger.pooled_sr_variance(ledger_path)
    if pooled_variance is not None:
        variance = pooled_variance
    else:
        sharpes = [float(item["stats_in_sample"]["sharpe"]) for item in ordered
                   if math.isfinite(float(item["stats_in_sample"]["sharpe"]))]
        variance = float(np.var(sharpes, ddof=1)) if len(sharpes) >= 2 else math.nan
        if not math.isfinite(variance) or variance <= 0:
            variance = float(constants.DEFAULT_SR_VARIANCE_FLOOR)
    return {"n_this": n_this, "post_batch_ledger_floor": post_batch_floor,
            "n_trials": shared_n, "sr_variance": variance,
            "nondegenerate_series_count": len(nondegenerate),
            "computed_family_count": len(ordered)}


def _instrument_set_for(item: dict) -> list:
    """The evidence schema wants a LIST of real tickers.

    MUST read `item['tickers']` (the real instrument list every producer
    that reuses this scoring logic carries -- this script's own
    compute_candidate, plus run_swing_mining.py/run_day_trading_mining.py,
    which import _score_one and _record_defensive_outcomes UNMODIFIED),
    NEVER `item['ticker']`: that field is a DISPLAY
    string and can be an arbitrary human-readable label (run_swing_mining.
    py's own `_ticker_label` falls back to `candidate.get('ticker_label')`,
    e.g. "S&P 500 pool") rather than a parseable, uppercase ticker --
    splitting THAT on "/" would fail evidence's own ticker-shape validation.
    Falls back to parsing `item['ticker']` only for a producer that does not
    carry `tickers` -- defense in depth; no in-repo caller relies on it."""
    tickers = item.get("tickers")
    if tickers:
        return list(tickers)
    return [t for t in (item.get("ticker") or "").split("/") if t]


def _score_one(item: dict, correction: dict) -> dict:
    stats = item["stats_in_sample"]
    if stats["trade_count"] < constants.MIN_TRADES_FOR_SCORING:
        stage, value = "insufficient_evidence", None
        reason = (f"only {stats['trade_count']} in-sample trades, below "
                  f"MIN_TRADES_FOR_SCORING ({constants.MIN_TRADES_FOR_SCORING})")
        # `raw_catalog_parameters`, NEVER `catalog_parameters`: the latter
        # is compute_candidate's
        # own factory-ADAPTER view (renamed/dropped/cast keys per family),
        # not the real catalog dict -- recording it would misstate the
        # tested hypothesis and later make validate_retry_request falsely
        # report drift against an unchanged real catalog entry.
        catalog_entry = {"family_id": item["family_id"],
                         "parameters": dict(item.get("raw_catalog_parameters") or {})}
        # evidence.build_insufficient_trades reads stats['sharpe_in_sample'];
        # this pipeline's own stats dicts key the identical per-trade value
        # as 'sharpe' (see _trade_stats/pooling.pooled_trade_stats) --
        # translated here so a real measured value is recorded instead of
        # silently defaulting to None on every single payload.
        stats_for_evidence = {**stats, "sharpe_in_sample": stats.get("sharpe")}
        payload = evidence.build_insufficient_trades(
            stats=stats_for_evidence, catalog_entry=catalog_entry,
            instrument_set=_instrument_set_for(item),
            in_sample_start=item.get("in_sample_start"),
            in_sample_end=item.get("in_sample_end"),
            pinned_window={"holdout_start": item["holdout_start"],
                           "holdout_end": item["holdout_end"]},
            # holdout_net_edge_bps is computed UNCONDITIONALLY for every
            # item by compute_candidate (this file's own producer) and by
            # run_swing_mining.py/run_day_trading_mining.py's equivalents --
            # never gated on the trade-count check (see evidence.py's own
            # module docstring for the gated alternative).
            holdout_look="computed_not_scored",
            signal_source_path=_SIGNAL_SOURCE_PATH,
            signal_source_sha256=evidence.source_sha256(__file__),
            evidence_source="measured",
            reactivation={
                "kind": "none_available",
                "rationale": "no pre-registered widening/extension "
                             "predicate at scoring time; the family stays "
                             "open to a fresh look via a new family_id",
            })
    else:
        payload = None
        value = dsr.deflated_sharpe_ratio(
            sr=stats["sharpe"], n=stats["trade_count"], skew=stats["skew"],
            kurtosis=stats["kurtosis"], n_trials=correction["n_trials"],
            sr_variance=correction["sr_variance"])
        holdout_trade_count = item["stats_holdout"]["trade_count"]
        if value < constants.DSR_PASS_THRESHOLD:
            stage, reason = "cut_dsr", f"DSR {value:.4f}"
        elif holdout_trade_count < constants.MIN_HOLDOUT_TRADES:
            # The holdout sign check is a bare pass/fail with no minimum
            # sample size of its own, so without this rule a single lucky
            # holdout trade could carry a family through. A family that
            # reaches here has
            # already cleared the in-sample DSR bar (dsr_value above is
            # real), so this is a DEFERRAL, not a rejection: it stays
            # eligible, and the real in-sample figures are recorded inside
            # the evidence payload rather than discarded.
            stage, reason = "insufficient_evidence", (
                f"only {holdout_trade_count} holdout trades, below "
                f"MIN_HOLDOUT_TRADES ({constants.MIN_HOLDOUT_TRADES})")
            catalog_entry = {"family_id": item["family_id"],
                             "parameters": dict(item.get("raw_catalog_parameters") or {})}
            payload = evidence.build_insufficient_holdout_trades(
                in_sample_stats={**stats, "sharpe_in_sample": stats.get("sharpe")},
                holdout_stats=item["stats_holdout"],
                catalog_entry=catalog_entry,
                instrument_set=_instrument_set_for(item),
                in_sample_start=item.get("in_sample_start"),
                in_sample_end=item.get("in_sample_end"),
                pinned_window={"holdout_start": item["holdout_start"],
                               "holdout_end": item["holdout_end"]},
                holdout_look="computed_not_scored",
                signal_source_path=_SIGNAL_SOURCE_PATH,
                signal_source_sha256=evidence.source_sha256(__file__),
                evidence_source="measured",
                reactivation={
                    "kind": "none_available",
                    "rationale": "the holdout window is pinned and "
                                 "fixed-length; waiting cannot add holdout "
                                 "trades. The family stays open to a fresh "
                                 "look via a new family_id",
                })
            value = None
        elif backtest_engine.holdout_passes(
                stats["net_edge_bps"], item["holdout_net_edge_bps"]):
            stage, reason = "survived", (
                f"holdout edge {item['holdout_net_edge_bps']:.1f} bps")
        else:
            stage, reason = "cut_holdout", (
                f"holdout edge {item['holdout_net_edge_bps']:.1f} bps")
    # sharpe_in_sample carried only when a real DSR was computed (dsr.py's
    # value is not None) -- see record_outcome's own docstring on why a
    # fabricated sharpe on an unscored outcome would corrupt the pool.
    sharpe_in_sample = stats["sharpe"] if value is not None else None
    return {**item, "stage": stage, "dsr_value": value, "reason": reason,
            "sharpe_in_sample": sharpe_in_sample, "evidence": payload,
            "shared_n_trials": correction["n_trials"],
            "shared_sr_variance": correction["sr_variance"]}


def _record_defensive_outcomes(ledger_path, newly_pinned, outcomed, *,
                               run_id, tested_date, error):
    """A caught mechanical failure (bad data shape, a raised exception
    partway through a batch) is `not_evaluated_error`, never `cut_dsr`:
    a bug is not a statistical
    finding about the market, and evidence.py's own
    build_engine_error/record_outcome biconditional makes writing it as a
    SCORED outcome with no dsr_value impossible -- labelling it
    not_evaluated_error here is correct at the source, not a workaround for
    that check."""
    for item in newly_pinned:
        family_id = item["family_id"]
        if family_id in outcomed:
            continue
        payload = evidence.build_engine_error(
            error=error, instrument_set=_instrument_set_for(item),
            pinned_window={"holdout_start": item["holdout_start"],
                           "holdout_end": item["holdout_end"]},
            # Same rationale as _score_one's own insufficient_evidence path:
            # holdout_net_edge_bps is computed unconditionally, before any
            # min-trades gate, by every producer that reaches this function.
            holdout_look="computed_not_scored")
        trial_ledger.record_outcome(
            ledger_path, family_id, run_id, tested_date, "not_evaluated_error",
            dsr_value=None, evidence=payload,
            note=(f"Batch scoring raised {type(error).__name__}: {error}. "
                  "Recorded defensively after first_test."),
            ticker_label=item.get("ticker"))
        outcomed.add(family_id)


def score_batch(computed: list[dict], *, ledger_path: str, run_id: str,
                tested_date: str, dry_run: bool,
                correction: dict | None = None,
                retest_family_ids: frozenset = frozenset()) -> list[dict]:
    """Score one fully computed batch; mutation is a post-compute phase.

    ``retest_family_ids``: a family
    in this set that is ALREADY pinned is treated exactly like the
    not_evaluated_error retry path -- its computed window must match the
    pinned window (else RuntimeError, same message style), the idempotent
    ``record_first_test`` no-op (``False``) is expected rather than a race,
    and a fresh outcome record is appended. Purpose: re-run a listed family
    on a widened instrument pool under the SAME family_id and the SAME
    pinned holdout window, with NO new trial charge (post_batch_floor never
    re-charges an id already in the ledger). With the default (empty) set,
    an already-pinned family with a real outcome raises "already pinned".
    """
    retest_family_ids = frozenset(retest_family_ids or ())
    ordered = sorted(computed, key=lambda item: item["family_id"])
    incomplete = [item["family_id"] for item in ordered
                  if not item.get("holdout_complete", False)]
    if incomplete:
        raise RuntimeError(f"incomplete holdout data for {incomplete}")
    shared = correction or compute_shared_correction(ordered, ledger_path)
    newly_pinned, outcomed = [], set()
    if not dry_run:
        # A not_evaluated_error family is a caught DEFECT, not evidence
        # about the market (evidence.build_engine_error's own docstring) -- it
        # must be able to re-enter scoring once the defect is fixed,
        # exactly like ruled_out.filter_eligible already treats it at the
        # catalog-filter level. Every OTHER outcome (a real score, or
        # insufficient_evidence) means this family genuinely already has a
        # result, and the "already pinned" refusal stands.
        already_pinned_ids = {
            item["family_id"] for item in ordered
            if trial_ledger.has_been_tested(ledger_path, item["family_id"])}
        non_retryable = set()
        retryable_ids = set()
        for family_id in already_pinned_ids:
            # An explicitly-listed retest family (see the docstring) takes
            # the same guarded path as a not_evaluated_error retry; every
            # other already-pinned family with a real outcome still refuses.
            explicit_retest = family_id in retest_family_ids
            if (not explicit_retest and
                    trial_ledger.latest_outcome(ledger_path, family_id) != "not_evaluated_error"):
                non_retryable.add(family_id)
                continue
            # A not_evaluated_error retry is safe ONLY against the SAME
            # pinned window the earlier failed attempt used --
            # record_first_test is idempotent and will NOT re-pin, so
            # scoring against a DIFFERENT window here would silently
            # record an outcome that does not match what the ledger says
            # was tested. Fail loud instead of risking that.
            item = next(i for i in ordered if i["family_id"] == family_id)
            pinned = trial_ledger.get_pinned_holdout_window(ledger_path, family_id)
            if (pinned["holdout_start"] != item["holdout_start"] or
                    pinned["holdout_end"] != item["holdout_end"]):
                retry_kind = ("explicit retest" if explicit_retest
                              else "not_evaluated_error retry")
                raise RuntimeError(
                    f"{family_id}: {retry_kind} window mismatch -- "
                    f"pinned {pinned['holdout_start']}..{pinned['holdout_end']}, "
                    f"this run computed {item['holdout_start']}.."
                    f"{item['holdout_end']}. Refusing to score against a "
                    f"different window than the one the ledger says was "
                    f"tested.")
            retryable_ids.add(family_id)
        if non_retryable:
            raise RuntimeError(
                f"already pinned before batch pin loop: {sorted(non_retryable)}")
        pin_errors = []
        for item in ordered:
            family_id = item["family_id"]
            try:
                recorded = trial_ledger.record_first_test(
                    ledger_path, family_id, tested_date,
                    item["holdout_start"], item["holdout_end"], [], run_id)
                # `recorded is False` is expected, not an error, for a
                # verified not_evaluated_error retry (the window-match
                # check above already confirmed this is safe) -- the
                # window was already pinned by the earlier failed attempt,
                # and record_first_test correctly no-ops rather than
                # re-pinning. For any other family, False means something
                # pinned it between the snapshot above and this loop (a
                # genuine race), which still must raise.
                if not recorded and family_id not in retryable_ids:
                    raise RuntimeError("record_first_test returned False")
                # An EXPLICIT retest of an already-pinned family (recorded is
                # False) was NOT pinned by this run, so it must never receive
                # _record_defensive_outcomes' not_evaluated_error if scoring
                # later raises: that would flip its latest REAL outcome into a
                # defect deferral, and a later DEFAULT run would then re-score
                # it through the not_evaluated_error retry path, bypassing the
                # --retest rule entirely.
                # A not_evaluated_error retry, by contrast, IS appended, so a
                # second failure re-records not_evaluated_error.
                if recorded or family_id not in retest_family_ids:
                    newly_pinned.append(item)
            except Exception as exc:
                pin_errors.append((family_id, exc))
        if pin_errors:
            failure = RuntimeError("pin failures: " + "; ".join(
                f"{fid}: {type(exc).__name__}: {exc}" for fid, exc in pin_errors))
            _record_defensive_outcomes(
                ledger_path, newly_pinned, outcomed, run_id=run_id,
                tested_date=tested_date, error=failure)
            raise failure
    results = []
    try:
        for item in ordered:
            result = _score_one(item, shared)
            results.append(result)
            if not dry_run:
                trial_ledger.record_outcome(
                    ledger_path, item["family_id"], run_id, tested_date,
                    result["stage"], dsr_value=result["dsr_value"],
                    note=f"{item.get('note', '')} Result: {result['reason']}.",
                    ticker_label=item.get("ticker"),
                    sharpe_in_sample=result["sharpe_in_sample"],
                    evidence=result.get("evidence"))
                outcomed.add(item["family_id"])
        return results
    except Exception as exc:
        if not dry_run:
            _record_defensive_outcomes(
                ledger_path, newly_pinned, outcomed, run_id=run_id,
                tested_date=tested_date, error=exc)
        raise




def public_result(item: dict) -> dict:
    excluded = {"daily_returns", "in_sample_trades", "holdout_trades",
                "in_sample_index", "holdout_index"}
    public = {}
    for key, value in item.items():
        if key in excluded:
            continue
        if isinstance(value, np.generic):
            value = value.item()
        elif isinstance(value, pd.Timestamp):
            value = value.isoformat()
        public[key] = value
    return public


def _file_sha256(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: str, text: str) -> None:
    destination = os.path.abspath(path)
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=directory, prefix=f".{os.path.basename(destination)}.",
        suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    finally:
        try:
            os.remove(temporary)
        except OSError:
            pass


def regenerate_results_log(*, catalog_path: str, ledger_path: str,
                           research_root: str, out_path: str) -> None:
    entries = catalog.load_catalog(catalog_path)
    rendered = results_log.render_results_log(
        entries, ledger_path, research_root)
    _atomic_write_text(out_path, rendered)


def regenerate_near_miss_log(*, ledger_path: str, out_path: str) -> None:
    """Refresh NEAR-MISS-LOG.md the same way `regenerate_results_log`
    refreshes RESULTS-LOG.md -- read-only over the trial ledger, atomic
    write. Kept as its OWN function (not folded into
    `regenerate_results_log`) so the two durable artifacts can be wired
    into the SAME finalization call sites without coupling their
    implementations, and each report keeps a single responsibility.
    Unlike `regenerate_results_log`, this needs no `catalog_path`/
    `research_root` -- the near-miss report reads only the ledger."""
    rendered = near_miss.render_near_miss_log(ledger_path)
    _atomic_write_text(out_path, rendered)


def run_batch(scope: dict, *, tested_date: str, run_id: str,
              ledger_path: str, dry_run: bool,
              ohlc_loader: Callable[[str], pd.DataFrame] = _load_ohlc,
              finra_loader: Callable = external_data.fetch_finra_margin_debt) -> dict:
    batch = compute_batch(
        scope, tested_date=tested_date, ohlc_loader=ohlc_loader,
        finra_loader=finra_loader)
    correction = compute_shared_correction(batch["computed"], ledger_path)
    scored = score_batch(
        batch["computed"], ledger_path=ledger_path, run_id=run_id,
        tested_date=tested_date, dry_run=dry_run, correction=correction)
    by_id = {item["family_id"]: item for item in scored}
    reconciliation = []
    for row in batch["reconciliation"]:
        item = by_id.get(row["family_id"])
        if item is None:
            reconciliation.append(row)
        else:
            reconciliation.append({**row, "status": "tested",
                                   "stage": item["stage"],
                                   "dsr_value": item["dsr_value"],
                                   "reason": item["reason"]})
    return {"schema_version": 1, "run_id": run_id,
            "tested_date": tested_date, "dry_run": dry_run,
            **batch["window"], "shared_correction": correction,
            "results": [public_result(item) for item in scored],
            "reconciliation": reconciliation,
            "gate_exclusions": scope["gate_exclusions"],
            "input_coverage": batch["input_coverage"],
            "input_errors": batch["input_errors"],
             "external_metadata": batch["external_metadata"]}


def execute_batch_run(
        scope: dict, *, tested_date: str, run_id: str, ledger_path: str,
        dry_run: bool, output_path: str, catalog_path: str,
        research_root: str, results_log_path: str,
        near_miss_log_path: str | None = None,
        batch_runner: Callable | None = None,
        results_renderer: Callable | None = None,
        near_miss_renderer: Callable | None = None) -> dict:
    """Run once and refresh durable evidence after every real ledger change.

    ``score_batch`` guarantees that a post-pin exception defensively closes
    every new pin. This outer ``finally`` closes the second half of the
    pipeline contract: any non-dry ledger mutation also regenerates the
    cumulative results log -- AND the near-miss log -- before success or failure escapes the
    process. Both refresh at the SAME point under the SAME
    `not dry_run and ledger changed` gate, so neither can go stale relative
    to the other. `near_miss_log_path` defaults to a sibling of
    `results_log_path` (same directory, `NEAR-MISS-LOG.md`) when omitted, so
    a call site that sets only `results_renderer`/`results_log_path` keeps
    both logs refreshing.
    """
    runner = batch_runner or run_batch
    renderer = results_renderer or regenerate_results_log
    nm_renderer = near_miss_renderer or regenerate_near_miss_log
    nm_log_path = near_miss_log_path or os.path.join(
        os.path.dirname(results_log_path) or ".", "NEAR-MISS-LOG.md")
    ledger_before = _file_sha256(ledger_path)
    def refresh_durable_evidence() -> None:
        ledger_after = _file_sha256(ledger_path)
        if not dry_run and ledger_after != ledger_before:
            renderer(
                catalog_path=catalog_path, ledger_path=ledger_path,
                research_root=research_root, out_path=results_log_path)
            nm_renderer(ledger_path=ledger_path, out_path=nm_log_path)

    try:
        payload = runner(
            scope, tested_date=tested_date, run_id=run_id,
            ledger_path=ledger_path, dry_run=dry_run)
        _atomic_write_text(
            output_path,
            json.dumps(payload, indent=2, allow_nan=False) + "\n")
    except BaseException as run_error:
        try:
            refresh_durable_evidence()
        except BaseException as regeneration_error:
            group_type = (
                ExceptionGroup
                if isinstance(run_error, Exception)
                and isinstance(regeneration_error, Exception)
                else BaseExceptionGroup)
            raise group_type(
                "batch run and results-log regeneration both failed",
                [run_error, regeneration_error]) from None
        raise
    refresh_durable_evidence()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", default="batch3")
    parser.add_argument("--tested-date",
                        default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--ledger", default=LEDGER_PATH)
    parser.add_argument("--catalog", default=CATALOG_PATH)
    parser.add_argument("--matrix", default=MATRIX_PATH)
    parser.add_argument("--out")
    parser.add_argument("--research-root", default=RESEARCH_ROOT)
    parser.add_argument("--results-log-out", default=RESULTS_LOG_PATH)
    parser.add_argument("--near-miss-log-out", default=NEAR_MISS_LOG_PATH)
    args = parser.parse_args()
    output_path = args.out or (DRY_RUN_PATH if args.dry_run else REAL_RUN_PATH)
    scope = load_run_scope(args.catalog, args.matrix, args.ledger)
    payload = execute_batch_run(
        scope, tested_date=args.tested_date, run_id=args.run_id,
        ledger_path=args.ledger, dry_run=args.dry_run,
        output_path=output_path, catalog_path=args.catalog,
        research_root=args.research_root,
        results_log_path=args.results_log_out,
        near_miss_log_path=args.near_miss_log_out)
    print(json.dumps({
        "output": output_path,
        "tested": sum(row["status"] == "tested"
                      for row in payload["reconciliation"]),
        "skipped": sum(row["status"] == "skipped"
                       for row in payload["reconciliation"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
