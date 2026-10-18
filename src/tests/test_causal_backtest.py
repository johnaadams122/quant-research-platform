import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

import hmm_guards
from _synth import two_regime_returns, prices_from_returns
from causal_backtest import cash_log_returns_by_date

PARAMS = {"means": [-0.008, 0.0008], "variances": [0.0009, 0.0001],
          "transmat": [[0.97, 0.03], [0.02, 0.98]], "startprob": [0.5, 0.5]}


def _write_irx(tmp, dates, rates):
    pd.DataFrame({"Date": pd.to_datetime(dates),
                  "irx_rate_annual": rates}).to_csv(
        os.path.join(str(tmp), "IRX.csv"), index=False)


def test_cash_rate_uses_prior_known_rate(tmp_path):
    _write_irx(tmp_path, ["2020-01-02", "2020-01-03", "2020-01-06"],
               [0.05, 0.10, 0.20])
    fd = pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07"]).date
    s = cash_log_returns_by_date(fd, str(tmp_path))
    assert abs(s.iloc[0] - np.log1p(0.05 / 252)) < 1e-12   # NOT 0.10
    assert abs(s.iloc[1] - np.log1p(0.10 / 252)) < 1e-12
    assert abs(s.iloc[2] - np.log1p(0.20 / 252)) < 1e-12   # ffill, no backfill


def test_cash_no_backfill_leading_nan_is_zero(tmp_path):
    _write_irx(tmp_path, ["2020-01-06"], [0.20])
    fd = pd.to_datetime(["2020-01-02", "2020-01-07"]).date   # 01-02 has no prior
    s = cash_log_returns_by_date(fd, str(tmp_path))
    assert s.iloc[0] == 0.0 or np.isnan(s.iloc[0])   # leading gap -> 0 after fillna


def test_path_terminal_matches_shipped_filter():
    from causal_backtest import forward_filter_pbull_path
    r = two_regime_returns(seed=11)
    path = forward_filter_pbull_path(PARAMS, r)
    assert path.shape == (len(r),)
    assert np.all((path >= 0.0) & (path <= 1.0))
    assert path[-1] == pytest.approx(
        hmm_guards.forward_filter_pbull(PARAMS, r), abs=1e-12)


def test_every_prefix_matches_shipped_filter():
    from causal_backtest import forward_filter_pbull_path
    r = two_regime_returns(n_bull=40, n_bear=20, n_bull2=20, seed=5)
    path = forward_filter_pbull_path(PARAMS, r)
    for k in (1, 7, 33, len(r)):
        assert path[k - 1] == pytest.approx(
            hmm_guards.forward_filter_pbull(PARAMS, r[:k]), abs=1e-12)


def test_empty_input_raises():
    from causal_backtest import forward_filter_pbull_path
    with pytest.raises(ValueError):
        forward_filter_pbull_path(PARAMS, np.array([]))


def test_trailing_ratio_matches_current_regime_tail():
    from causal_backtest import trailing_ratio_path
    from _synth import prices_from_returns
    df = prices_from_returns(two_regime_returns(seed=9))
    path = trailing_ratio_path(df, "PX", vol_window=63)
    assert len(path) == len(df) - 1
    assert np.isnan(path.iloc[:62]).all()
    lr = np.log1p(df["PX"].pct_change()).dropna()
    tail = lr.iloc[-63:]
    expected = float(tail.mean() * 252) / (2 * float(tail.var() * 252))
    assert path.iloc[-1] == pytest.approx(expected, rel=1e-12)


def test_trailing_ratio_nan_on_zero_variance():
    from causal_backtest import trailing_ratio_path
    import pandas as pd
    dates = pd.bdate_range("2020-01-01", periods=80)
    df = pd.DataFrame({"Date": dates.date, "PX": 100.0})   # flat prices
    path = trailing_ratio_path(df, "PX", vol_window=63)
    assert np.isnan(path.iloc[-1])


def _small_path(df, **kw):
    from causal_backtest import causal_pbull_path
    defaults = dict(refit_every=10, warmup=60, n_restarts=2)
    defaults.update(kw)
    return causal_pbull_path(df, "PX", **defaults)


def _sane_always(monkeypatch):
    import causal_backtest as cb
    monkeypatch.setattr(cb.hmm_guards, "fit_is_sane",
                        lambda model, states: (True, []))


def test_truncation_invariance(monkeypatch):
    # Sanity verdicts SCRIPTED sane: the invariance property is about the
    # refit-schedule anchoring, not real fit quality. On this toy fixture
    # the REAL fit_is_sane rejects every fit before bar
    # 110 (the warmup window is single-regime), which would silently shrink
    # the compared prefix.
    from _synth import prices_from_returns
    _sane_always(monkeypatch)
    df = prices_from_returns(
        two_regime_returns(n_bull=80, n_bear=40, n_bull2=40, seed=3))
    full = _small_path(df)
    k = 120                                   # price rows; lr bars = k-1
    trunc = _small_path(df.iloc[:k])
    m = len(trunc)
    assert m > 0
    pd.testing.assert_frame_equal(full.iloc[:m].reset_index(drop=True),
                                  trunc.reset_index(drop=True))


def test_path_is_causal_shape_and_range(monkeypatch):
    from _synth import prices_from_returns
    _sane_always(monkeypatch)                  # see note above
    df = prices_from_returns(
        two_regime_returns(n_bull=80, n_bear=40, n_bull2=40, seed=3))
    path = _small_path(df)
    n_lr = len(df) - 1
    assert len(path) == n_lr - 60             # warmup bars excluded
    assert path["p_bull"].between(0, 1).all()
    assert (np.diff(path["refit_bar"].unique()) >= 10).all()


def test_insane_refit_keeps_previous_params(monkeypatch):
    # Verdicts FULLY SCRIPTED (never "real except call #2"): the contract
    # must not depend on what the real HMM finds sane on a toy fixture
    # (on this fixture it rejects the first several fits).
    from _synth import prices_from_returns
    import causal_backtest as cb
    df = prices_from_returns(
        two_regime_returns(n_bull=80, n_bear=40, n_bull2=40, seed=3))
    verdicts = iter([(True, []), (False, ["means_too_close"])])
    monkeypatch.setattr(cb.hmm_guards, "fit_is_sane",
                        lambda model, states: next(verdicts, (True, [])))
    path = _small_path(df)
    bars = sorted(path["refit_bar"].unique())
    # refit at bar 70 was rejected -> bars 70-79 still carry bar-60 params
    assert 60 in bars and 70 not in bars and 80 in bars


def test_first_insane_refits_drop_leading_rows(monkeypatch):
    # Verdicts FULLY SCRIPTED: the first scheduled refit (bar 60) is rejected,
    # so its whole segment is dropped; the walk-forward finds the first sane
    # fit at bar 70 and rows begin there.
    from _synth import prices_from_returns
    import causal_backtest as cb
    df = prices_from_returns(
        two_regime_returns(n_bull=80, n_bear=40, n_bull2=40, seed=3))
    verdicts = iter([(False, ["means_too_close"])])
    monkeypatch.setattr(cb.hmm_guards, "fit_is_sane",
                        lambda model, states: next(verdicts, (True, [])))
    path = _small_path(df)
    n_lr = len(df) - 1
    assert path["refit_bar"].min() == 70      # bar-60 segment dropped entirely
    assert len(path) == n_lr - 70


def _fake_corpus(tmp_path, seed=3):
    from _synth import prices_from_returns
    df = prices_from_returns(
        two_regime_returns(n_bull=80, n_bear=40, n_bull2=40, seed=seed),
        ticker="QQQ")
    syn = df.rename(columns={"QQQ": "QQQ_3x_synthetic"}).copy()
    syn["QQQ_3x_synthetic"] = syn["QQQ_3x_synthetic"] ** 1.5   # any distinct series
    raw = tmp_path / "raw"; raw.mkdir()
    df.to_csv(raw / "QQQ.csv", index=False)
    syn.to_csv(raw / "QQQ_3x_synthetic.csv", index=False)
    # IRX.csv covering every price date -> no leading-NaN gap in lr_cash
    _write_irx(raw, df["Date"], [0.05] * len(df))
    return str(raw)


def test_joint_frame_shape_and_cache_roundtrip(tmp_path):
    from causal_backtest import build_joint_frame
    raw = _fake_corpus(tmp_path)
    cache = str(tmp_path / "backtest")
    j1 = build_joint_frame("QQQ", 10, raw, cache, vol_window=20,
                           warmup=60, n_restarts=2)
    assert list(j1.columns) == ["Date", "p_bull", "trail_ratio",
                                "lr_1x", "lr_3x", "lr_cash"]
    assert len(j1) > 0 and j1["p_bull"].between(0, 1).all()
    import causal_backtest as cb, pytest as _pt
    def boom(*a, **k): raise AssertionError("cache miss - HMM refit ran")
    orig = cb.causal_pbull_path
    cb.causal_pbull_path = boom
    try:
        j2 = build_joint_frame("QQQ", 10, raw, cache, vol_window=20,
                               warmup=60, n_restarts=2)
    finally:
        cb.causal_pbull_path = orig
    pd.testing.assert_frame_equal(
        j1.reset_index(drop=True), j2.reset_index(drop=True),
        check_exact=False, rtol=1e-12)


def test_meta_mismatch_rebuilds(tmp_path):
    from causal_backtest import build_joint_frame
    raw = _fake_corpus(tmp_path)
    cache = str(tmp_path / "backtest")
    build_joint_frame("QQQ", 10, raw, cache, vol_window=20,
                      warmup=60, n_restarts=2)
    import json as _json, os as _os
    meta_fp = _os.path.join(cache, "QQQ_re10.meta.json")
    meta = _json.load(open(meta_fp))
    meta["corpus_max_date"] = "1990-01-01"
    _json.dump(meta, open(meta_fp, "w"))
    called = {"n": 0}
    import causal_backtest as cb
    orig = cb.causal_pbull_path
    def spy(*a, **k):
        called["n"] += 1
        return orig(*a, **k)
    cb.causal_pbull_path = spy
    try:
        build_joint_frame("QQQ", 10, raw, cache, vol_window=20,
                          warmup=60, n_restarts=2)
    finally:
        cb.causal_pbull_path = orig
    assert called["n"] == 1


def test_corrupt_cache_date_column_rebuilds(tmp_path):
    from causal_backtest import build_joint_frame
    raw = _fake_corpus(tmp_path)
    cache = str(tmp_path / "backtest")
    build_joint_frame("QQQ", 10, raw, cache, vol_window=20,
                      warmup=60, n_restarts=2)
    # corrupt ONE date value: read_csv(parse_dates=...) keeps strings, .dt dies
    import os as _os
    fp = _os.path.join(cache, "QQQ_re10.csv")
    lines = open(fp, encoding="utf-8").read().splitlines()
    parts = lines[1].split(",")
    parts[0] = "not-a-date"
    lines[1] = ",".join(parts)
    with open(fp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    j = build_joint_frame("QQQ", 10, raw, cache, vol_window=20,
                          warmup=60, n_restarts=2)   # must rebuild, not crash
    assert len(j) > 0 and j["p_bull"].between(0, 1).all()


def test_nan_trail_ratio_rows_are_kept(tmp_path, monkeypatch):
    # vol_window > warmup => the causal path's first rows predate a full
    # rolling window -> trail_ratio NaN there, and those rows must survive.
    # Sanity verdicts SCRIPTED sane (same reason as test_truncation_invariance:
    # the REAL fit_is_sane rejects every fit before bar 110 on this fixture,
    # which starts the path past the vol_window=70 NaN region and would make
    # this test vacuously green-or-fail on fit behavior, not on the KEPT rule).
    from causal_backtest import build_joint_frame
    _sane_always(monkeypatch)
    raw = _fake_corpus(tmp_path)
    cache = str(tmp_path / "backtest2")
    j = build_joint_frame("QQQ", 10, raw, cache, vol_window=70,
                          warmup=60, n_restarts=2)
    assert j["trail_ratio"].isna().any()
    nan_rows = j[j["trail_ratio"].isna()]
    assert nan_rows["lr_1x"].notna().all() and nan_rows["lr_3x"].notna().all()


def _spy_rebuild(raw, cache):
    """Re-call build_joint_frame with a spy on causal_pbull_path; returns
    (frame, times_called)."""
    import causal_backtest as cb
    from causal_backtest import build_joint_frame
    called = {"n": 0}
    orig = cb.causal_pbull_path
    def spy(*a, **k):
        called["n"] += 1
        return orig(*a, **k)
    cb.causal_pbull_path = spy
    try:
        j = build_joint_frame("QQQ", 10, raw, cache, vol_window=20,
                              warmup=60, n_restarts=2)
    finally:
        cb.causal_pbull_path = orig
    return j, called["n"]


def test_cache_rebuilds_when_source_content_changes_same_dates(tmp_path):
    # a regenerated source CSV (e.g. a changed financing input in the synthetic)
    # with the SAME last date must invalidate the cache -- corpus_max_date
    # alone cannot see it; the meta content fingerprints must
    from causal_backtest import build_joint_frame
    raw = _fake_corpus(tmp_path)
    cache = str(tmp_path / "backtest")
    build_joint_frame("QQQ", 10, raw, cache, vol_window=20,
                      warmup=60, n_restarts=2)
    import os as _os
    syn_fp = _os.path.join(raw, "QQQ_3x_synthetic.csv")
    syn = pd.read_csv(syn_fp)
    syn["QQQ_3x_synthetic"] = syn["QQQ_3x_synthetic"] * 1.01   # same dates
    syn.to_csv(syn_fp, index=False)
    j, n = _spy_rebuild(raw, cache)
    assert n == 1                              # rebuilt, not stale cache
    assert len(j) > 0


def test_cache_schema_broken_or_empty_rebuilds(tmp_path):
    # The header-only (0-row) cache is one broken case. NOTE: without the
    # schema check that case would still rebuild BY ACCIDENT -- pandas
    # leaves an empty object-dtype Date column, .dt.date raises
    # AttributeError into the fail-safe except. The discriminating case is
    # phase 1: a cache with valid rows but a MISSING column reads back
    # silently and, without the check, would be returned as-is.
    from causal_backtest import build_joint_frame
    raw = _fake_corpus(tmp_path)
    cache = str(tmp_path / "backtest")
    build_joint_frame("QQQ", 10, raw, cache, vol_window=20,
                      warmup=60, n_restarts=2)
    import os as _os
    fp = _os.path.join(cache, "QQQ_re10.csv")

    # phase 1: schema-broken cache (valid rows, missing column) -> rebuild
    pd.read_csv(fp).drop(columns=["trail_ratio"]).to_csv(fp, index=False)
    j2, n2 = _spy_rebuild(raw, cache)
    assert n2 == 1
    assert list(j2.columns) == ["Date", "p_bull", "trail_ratio",
                                "lr_1x", "lr_3x", "lr_cash"] and len(j2) > 0

    # phase 2: header-only (0-row, correct header) cache -> rebuild,
    # never return an empty frame
    with open(fp, encoding="utf-8") as f:
        header = f.readline()
    with open(fp, "w", encoding="utf-8") as f:
        f.write(header)
    j3, n3 = _spy_rebuild(raw, cache)
    assert n3 == 1 and len(j3) > 0


# --- parallel+resumable sweep: lossless round_trip read + strict ---

_NUMERIC_COLS = ("p_bull", "trail_ratio", "lr_1x", "lr_3x", "lr_cash")


def test_build_joint_frame_reload_is_bytewise_in_memory(tmp_path, monkeypatch):
    """Linchpin, through the REAL function: the causal frame built
    in-memory (force=True return) must equal a fresh force=False reload of the
    cache it just wrote, byte-for-byte on every numeric float64 column, on a
    NaN-bearing corpus. A default-precision CSV read can drift in the last
    bits and fail the tobytes() comparison; the round_trip read recovers it
    exactly.

    Sanity verdicts SCRIPTED sane (same reason as test_nan_trail_ratio_rows_
    are_kept: the real fit_is_sane rejects every fit before ~bar 110 on this
    toy fixture, which would start the path past the vol_window NaN region)."""
    from causal_backtest import build_joint_frame
    _sane_always(monkeypatch)
    raw = _fake_corpus(tmp_path)
    cache = str(tmp_path / "backtest")
    # vol_window(70) > warmup(60) -> the path's leading rows predate a full
    # rolling window, so trail_ratio carries real NaN that must round-trip too.
    j_mem = build_joint_frame("QQQ", 10, raw, cache, vol_window=70,
                              warmup=60, n_restarts=2, force=True)
    assert j_mem["trail_ratio"].isna().any()        # NaN actually present
    j_reload = build_joint_frame("QQQ", 10, raw, cache, vol_window=70,
                                 warmup=60, n_restarts=2, force=False)
    for c in _NUMERIC_COLS:
        a = j_mem[c].to_numpy(dtype="<f8")
        b = j_reload[c].to_numpy(dtype="<f8")
        assert a.tobytes() == b.tobytes(), f"{c} reload != in-memory (bytewise)"
    # Date is exact regardless of float precision; compare as values.
    assert list(j_mem["Date"]) == list(j_reload["Date"])


def test_build_joint_frame_strict_raises_without_fitting(tmp_path, monkeypatch):
    """With a VALID raw corpus but an ABSENT causal cache, strict=True must
    raise the strict-guard FileNotFoundError WITHOUT invoking any HMM code
    (causal_pbull_path is fail-if-called). The corpus is real so the guard is
    the only thing that can raise: with a nonexistent data_dir, load_series
    would raise before the guard ever ran and the test would pass vacuously."""
    import causal_backtest as cb
    from causal_backtest import build_joint_frame
    raw = _fake_corpus(tmp_path)
    cache = str(tmp_path / "empty_cache")        # no causal cache present
    called = {"fit": 0}

    def boom(*a, **k):
        called["fit"] += 1
        raise AssertionError("causal_pbull_path ran under strict=True")
    monkeypatch.setattr(cb, "causal_pbull_path", boom)

    with pytest.raises(FileNotFoundError, match="strict=True"):
        build_joint_frame("QQQ", 10, raw, cache, vol_window=20, warmup=60,
                          n_restarts=2, strict=True)
    assert called["fit"] == 0                     # zero HMM fits


def test_build_joint_frame_strict_returns_valid_cache(tmp_path):
    """strict=True still RETURNS a present, meta-matching cache (only a miss
    raises). Guards against a guard that raises unconditionally."""
    from causal_backtest import build_joint_frame
    raw = _fake_corpus(tmp_path)
    cache = str(tmp_path / "backtest")
    j1 = build_joint_frame("QQQ", 10, raw, cache, vol_window=20, warmup=60,
                           n_restarts=2)                       # populate cache
    j2 = build_joint_frame("QQQ", 10, raw, cache, vol_window=20, warmup=60,
                           n_restarts=2, strict=True)          # hit under strict
    for c in _NUMERIC_COLS:
        assert (j1[c].to_numpy(dtype="<f8").tobytes()
                == j2[c].to_numpy(dtype="<f8").tobytes())
