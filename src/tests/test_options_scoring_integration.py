import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from mining import abi1_driver, options_curve, wheel_engine, evidence


def _synthetic_frame(n, price0=100.0, price1=115.0):
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    close = np.linspace(price0, price1, n)
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close},
                        index=idx)


def test_pooled_options_family_scores_through_the_real_abi1_pipeline(tmp_path):
    n = 260
    price = pd.Series(np.linspace(100.0, 118.0, n),
                      index=pd.date_range("2020-01-02", periods=n, freq="B"))
    iv = pd.Series(0.22, index=price.index)
    r = pd.Series(0.04, index=price.index)

    curves = {}
    for ticker, shift in (("SPY", 0.0), ("QQQ", 5.0)):
        cycle = wheel_engine.run_wheel(
            price + shift, iv, r, div_yield=0.0, target_delta=0.15, dte=21,
            profit_target_pct=0.5, cost_pct_of_premium=0.02)
        curves[ticker] = options_curve.build_options_curve(cycle)

    pooled_curve = options_curve.pool_options_curves(curves)
    benchmark_frame = _synthetic_frame(n, 100.0, 112.0)

    result = abi1_driver.evaluate_family(
        pooled_curve, benchmark_frame=benchmark_frame, benchmark_entry_cost_bps=0.5,
        account_type="taxable", tax_policy="none")
    assert result["verdict"] in {"not_scoreable", "cut_dsr", "cut_risk_refusal",
                                 "holdout_not_evaluable", "cut_holdout", "survived"}

    ledger_path = str(tmp_path / "trial_ledger.jsonl")
    catalog_entry = {
        "family_id": "options_wheel_pooled_test", "name": "test", "mechanism": "test",
        "tickers": ["SPY", "QQQ"], "parameters": {}, "account_types": ["taxable"],
        "status": "eligible", "source": "test", "added_date": "2026-09-10",
    }
    stage = abi1_driver.record_abi1_outcome(
        ledger_path, "options_wheel_pooled_test", "test-run", "2026-09-10", result,
        catalog_entry=catalog_entry, instrument_set=["SPY", "QQQ"],
        cash_convention="sweep_cash_daily", benchmark="SPY", tax_policy="none",
        ticker_label="pool-of-2",
        signal_source_path="src/mining/wheel_engine.py",
        signal_source_sha256=evidence.source_sha256(
            os.path.join(os.path.dirname(__file__), "..", "mining", "wheel_engine.py")))
    assert stage in {"survived", "cut_dsr", "cut_holdout", "cut_risk_refusal",
                     "insufficient_evidence"}
    assert os.path.exists(ledger_path)
