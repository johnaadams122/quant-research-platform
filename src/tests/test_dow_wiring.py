"""Dow (DIA) wiring into the sweep universe.

DIA joins strategy_sweep.TICKERS as the 4th sweep ticker. The causal seam is
a LOCAL entry in causal_backtest.UND_TO_SYN: regime_signal.LEV_MAP / SYN_MAP
stay byte-identical (the regime signal covers TQQQ/SOXL/SPXL only -- DIA is
a sweep-only research ticker).
"""
import pytest


def test_tickers_include_dia_as_fourth():
    import strategy_sweep as ss
    assert ss.TICKERS == ("QQQ", "SOX", "SPY", "DIA")


def test_und_to_syn_dia_is_local_and_live_maps_untouched():
    import causal_backtest as cb
    import regime_signal as rs
    # the sweep-side seam: DIA -> its synthetic 3x series
    assert cb.UND_TO_SYN["DIA"] == "DIA_3x_synthetic"
    # the derived trio is unchanged
    assert cb.UND_TO_SYN["QQQ"] == "QQQ_3x_synthetic"
    assert cb.UND_TO_SYN["SOX"] == "SOX_3x_synthetic"
    assert cb.UND_TO_SYN["SPY"] == "SPY_3x_synthetic"
    # separability invariant: regime_signal's maps are EXACTLY the three
    # leveraged legs -- no Dow entry may ever leak in via this wiring
    assert rs.LEV_MAP == {"TQQQ": "QQQ", "SOXL": "SOX", "SPXL": "SPY"}
    assert rs.SYN_MAP == {"TQQQ": "QQQ_3x_synthetic",
                          "SOXL": "SOX_3x_synthetic",
                          "SPXL": "SPY_3x_synthetic"}


def test_enumerate_specs_yields_24_units_with_dia():
    import strategy_sweep as ss
    import sweep_cache as sc
    bp = dict(n_boot=50, train_end="2014-12-31", n_restarts=1, vol_window=63,
              warmup=1000, data_dir="data/raw")
    specs = sc.enumerate_specs(ss.TICKERS, [0, 1, 2, 3, 4], "cache", bp)
    assert len(specs) == 24                    # 6 units x 4 tickers
    dia = [s for s in specs if s.ticker == "DIA"]
    assert sorted((s.block_len, s.seed) for s in dia) == [
        (63, 0), (63, 1), (63, 2), (63, 3), (63, 4), (126, 0)]


def test_effective_workers_cap_is_24(monkeypatch):
    import os
    import sweep_cache as sc
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    assert sc.effective_workers(999, free_gb=100.0) == 24
