# src/tests/test_data_pipeline_irx.py
import os, sys, pandas as pd, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_fetch_and_save_irx_sorts_and_dedups(tmp_path, monkeypatch):
    import data_pipeline as dp
    # unsorted, with a duplicate date -> must sort ascending and keep one row
    fake = pd.Series([0.052, 0.050, 0.051],
                     index=pd.to_datetime(["2020-01-06", "2020-01-02",
                                           "2020-01-06"]))
    monkeypatch.setattr(dp, "fetch_risk_free_rate", lambda start="max": fake)
    out = dp.fetch_and_save_irx(data_dir=str(tmp_path))
    disk = pd.read_csv(os.path.join(str(tmp_path), "IRX.csv"),
                       parse_dates=["Date"])
    assert list(disk.columns) == ["Date", "irx_rate_annual"]
    assert disk["Date"].is_monotonic_increasing
    assert len(disk) == 2                      # duplicate 01-06 collapsed
    assert disk["Date"].iloc[0] == pd.Timestamp("2020-01-02")

def test_fetch_and_save_irx_raises_on_empty(tmp_path, monkeypatch):
    import data_pipeline as dp
    monkeypatch.setattr(dp, "fetch_risk_free_rate",
                        lambda start="max": pd.Series([], dtype=float))
    with pytest.raises(RuntimeError):
        dp.fetch_and_save_irx(data_dir=str(tmp_path))

def test_run_pipeline_constant_fallback_skips_irx(monkeypatch):
    import data_pipeline as dp
    calls = {"irx": 0}
    monkeypatch.setattr(dp, "fetch_and_save_irx",
                        lambda *a, **k: calls.__setitem__("irx", calls["irx"] + 1))
    monkeypatch.setattr(dp, "fetch_risk_free_rate",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("network hit in constant-fallback mode")))
    dp.run_pipeline(tickers=[], synthetic_configs=[], use_market_rates=False)
    assert calls["irx"] == 0   # persistence must be skipped offline

def test_run_pipeline_market_rates_persists_irx(monkeypatch):
    import pandas as pd, data_pipeline as dp
    calls = {"irx": 0}
    monkeypatch.setattr(dp, "fetch_and_save_irx",
                        lambda *a, **k: calls.__setitem__("irx", calls["irx"] + 1))
    monkeypatch.setattr(dp, "fetch_risk_free_rate", lambda *a, **k: pd.Series([0.05]))
    dp.run_pipeline(tickers=[], synthetic_configs=[], use_market_rates=True)
    assert calls["irx"] == 1   # normal refresh persists IRX
