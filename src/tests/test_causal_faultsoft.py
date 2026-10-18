import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import hashlib, inspect
import numpy as np
import pandas as pd
import pytest
import causal_backtest as cb
from _synth import two_regime_returns, prices_from_returns

def _prices(n_bull=700, seed=0):
    # two_regime_returns(n_bull) -> n_bull + n_bear(150) + n_bull2(200) returns,
    # i.e. 1050 at n_bull=700. The two tail blocks are FIXED defaults, they do
    # NOT scale with n_bull -- so this is not "~2*n_bull".
    r = two_regime_returns(n_bull, seed=seed)
    return prices_from_returns(r, "QQQ")

def test_faultsoft_matches_oracle_bar_for_bar_on_clean_series():
    df = _prices()
    a = cb.causal_pbull_path(df, "QQQ", refit_every=21, warmup=1000, n_restarts=3)
    b = cb.causal_pbull_path_faultsoft(df, "QQQ", refit_every=21, warmup=1000,
                                       n_restarts=3)
    # Non-vacuity guard: every assertion below holds trivially on EMPTY frames
    # (which is what an all-refits-rejected or below-warmup tape produces), so
    # without this the whole test could silently pass while comparing nothing.
    assert len(a) > 0 and len(b) > 0
    assert list(a["Date"]) == list(b["Date"])
    assert np.array_equal(a["p_bull"].values, b["p_bull"].values)  # EXACT
    assert not b["degraded"].any()

def test_safe_forward_path_returns_valid_prefix_and_fail_index():
    # craft params + data that underflow partway: a huge outlier return at idx 5
    params = {"means": [-0.01, 0.001], "variances": [1e-6, 1e-6],
              "transmat": [[0.9, 0.1], [0.1, 0.9]], "startprob": [0.5, 0.5]}
    x = np.array([0.001, -0.001, 0.001, -0.001, 0.001, 50.0, 0.001])  # 50.0 underflows
    out, fail_idx = cb._safe_forward_path(params, x)
    assert fail_idx == 5                     # failed at the outlier bar
    assert out.shape[0] == 5                  # valid prefix only
    assert np.isfinite(out).all()

def test_faultsoft_holds_from_underflow_and_flags(monkeypatch):
    df = _prices()
    real_safe = cb._safe_forward_path
    def flaky(params, x):
        out, fi = real_safe(params, x)
        # force a failure 3 bars into any full-length pass to exercise the branch
        if x.size > 1010 and fi == x.size:
            return out[:1005], 1005
        return out, fi
    monkeypatch.setattr(cb, "_safe_forward_path", flaky)
    o = cb.causal_pbull_path_faultsoft(df, "QQQ", refit_every=21, warmup=1000,
                                       n_restarts=3)
    assert o["degraded"].any()
    assert o["p_bull"].between(0.0, 1.0).all()
    assert not o["p_bull"].isna().any()
    # a held-degraded bar equals the last non-degraded p_bull before it
    d = o.reset_index(drop=True)
    first_deg = d.index[d["degraded"]][0]
    assert d.loc[first_deg, "p_bull"] == d.loc[first_deg - 1, "p_bull"]

def test_existing_causal_functions_are_byte_identical():
    # golden-source guard: the ABI digest does NOT hash causal_backtest.py, so
    # pin the source of every pre-existing fn downstream code depends on.
    GOLDEN = {
        "causal_pbull_path": "a4ede62cce3ea11baa501215652e182564a1bde45b976c44eb68feb7b3b3edc8",
        "forward_filter_pbull_path": "8b615e0c08ef3ffe45c7ae00c46a5ed987ccdc6a331660357ec150bf215e65a1",
    }
    for name, want in GOLDEN.items():
        src = inspect.getsource(getattr(cb, name)).encode("utf-8")
        got = hashlib.sha256(src).hexdigest()
        assert got == want, f"{name} source changed (must stay byte-identical)"
