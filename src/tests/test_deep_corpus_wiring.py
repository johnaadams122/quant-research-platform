"""Deep index-owner corpora (NDX_TR / SPX_TR) wiring.

These are RESEARCH corpora sourced from the index owner (src/deep_corpus.py),
built because the QQQ/SPY ETF proxies track their own indices poorly in early
eras (QQQ vs XNDX annualized daily tracking error was 13.22% across 1999-2002 vs
1.07% post-2010). They enter the sweep ONLY via an explicit --tickers override,
so a default run is byte-for-byte unaffected by them.

Invariants pinned here:
  1. The default ticker universe is UNCHANGED (no silent scope creep).
  2. The causal seam is LOCAL to causal_backtest, exactly like the DIA wiring.
  3. regime_signal's signal maps stay byte-identical -- the advisory signal
     stays on TQQQ/SOXL/SPXL off the QQQ/SOX/SPY corpora, and no research
     corpus may ever leak into it.
  4. The compute-ABI digest matches its pinned golden value, so the deep
     wiring changes nothing the sweep cache keys on (the golden is the digest
     of the current hashed sources, with the interpreter and library versions
     held at the capture environment's; an intentional change to a hashed
     source, docstrings included, must re-pin it).
"""
import pytest

GOLDEN_ABI = "38f9081e6368ef3c14fd50de765b4ac729c951a5"


def test_default_ticker_universe_unchanged():
    """The deep corpora must NOT join the default universe."""
    import strategy_sweep as ss
    assert ss.TICKERS == ("QQQ", "SOX", "SPY", "DIA")
    assert "NDX_TR" not in ss.TICKERS and "SPX_TR" not in ss.TICKERS


def test_deep_seams_are_local_and_live_maps_untouched():
    import causal_backtest as cb
    import regime_signal as rs
    assert cb.UND_TO_SYN["NDX_TR"] == "NDX_TR_3x_synthetic"
    assert cb.UND_TO_SYN["SPX_TR"] == "SPX_TR_3x_synthetic"
    # pre-existing seams survive
    assert cb.UND_TO_SYN["QQQ"] == "QQQ_3x_synthetic"
    assert cb.UND_TO_SYN["SOX"] == "SOX_3x_synthetic"
    assert cb.UND_TO_SYN["SPY"] == "SPY_3x_synthetic"
    assert cb.UND_TO_SYN["DIA"] == "DIA_3x_synthetic"
    # signal-map separability: no research corpus may leak into the signal maps
    assert rs.LEV_MAP == {"TQQQ": "QQQ", "SOXL": "SOX", "SPXL": "SPY"}
    assert rs.SYN_MAP == {"TQQQ": "QQQ_3x_synthetic",
                          "SOXL": "SOX_3x_synthetic",
                          "SPXL": "SPY_3x_synthetic"}
    for m in (rs.LEV_MAP, rs.SYN_MAP):
        assert not any("NDX_TR" in str(v) or "SPX_TR" in str(v)
                       for v in list(m) + list(m.values()))


def test_tickers_flag_defaults_to_module_constant():
    """Omitting --tickers must resolve to the default module universe exactly."""
    import strategy_sweep as ss
    args = ss._parse_args([])
    assert args.tickers is None
    resolved = (tuple(t.strip() for t in args.tickers.split(",") if t.strip())
                if args.tickers else ss.TICKERS)
    assert resolved == ss.TICKERS


@pytest.mark.parametrize("raw,want", [
    ("NDX_TR,SPX_TR", ("NDX_TR", "SPX_TR")),
    (" NDX_TR , SPX_TR ", ("NDX_TR", "SPX_TR")),   # whitespace tolerated
    ("NDX_TR", ("NDX_TR",)),                        # single ticker
    ("NDX_TR,,SPX_TR", ("NDX_TR", "SPX_TR")),       # empty field dropped
])
def test_tickers_flag_parses(raw, want):
    import strategy_sweep as ss
    args = ss._parse_args(["--tickers", raw])
    resolved = tuple(t.strip() for t in args.tickers.split(",") if t.strip())
    assert resolved == want


def test_deep_corpus_configs_are_total_return():
    """XNDX and ^SP500TR are ALREADY total return -- a dividend add-back here
    would double-count. This is the SOX trap (price-return index) inverted."""
    import deep_corpus as dc
    assert {c["underlying"] for c in dc.DEEP_SYNTHETIC_CONFIGS} == {"NDX_TR", "SPX_TR"}
    for cfg in dc.DEEP_SYNTHETIC_CONFIGS:
        assert cfg["dividend_yield"] == 0.0, f"{cfg['underlying']}: double-counted dividends"
        assert cfg["leverage"] == 3.0


def test_nonpositive_closure_placeholder_is_dropped():
    """Nasdaq emits Value=0.0 for market closures (2012-10-29, Hurricane Sandy).
    log(0) -> -inf silently NaNs every downstream statistic."""
    import numpy as np
    import pandas as pd
    import deep_corpus as dc
    s = pd.Series([100.0, 0.0, 102.0],
                  index=pd.to_datetime(["2012-10-26", "2012-10-29", "2012-10-31"]))
    out = dc._drop_nonpositive(s, "TEST")
    assert len(out) == 2 and (out > 0).all()
    assert np.isfinite(np.log(out).diff().dropna()).all()


# The digest also hashes the interpreter and library versions, so the golden value
# is only reproducible with the versions it was captured under. Holding those four
# strings fixed lets this test check the CODE part on any machine; a real version
# change still invalidates cached sweep units in production, as intended.
GOLDEN_ABI_VERSIONS = {"py": "3.13.5", "np": "2.1.3", "pd": "2.2.3", "pa": "19.0.0"}


def test_abi_digest_unchanged_by_deep_wiring(monkeypatch):
    """The digest equals the pinned golden for the current hashed sources."""
    import sys
    import numpy
    import pandas
    import pyarrow
    import sweep_cache
    monkeypatch.setattr(sys, "version", GOLDEN_ABI_VERSIONS["py"] + " (golden)")
    monkeypatch.setattr(numpy, "__version__", GOLDEN_ABI_VERSIONS["np"])
    monkeypatch.setattr(pandas, "__version__", GOLDEN_ABI_VERSIONS["pd"])
    monkeypatch.setattr(pyarrow, "__version__", GOLDEN_ABI_VERSIONS["pa"])
    sweep_cache._ABI_MEMO.clear()
    try:
        assert sweep_cache.compute_abi_digest() == GOLDEN_ABI
    finally:
        sweep_cache._ABI_MEMO.clear()
