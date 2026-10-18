import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import hashlib
import inspect
import numpy as np
import pandas as pd
import pytest
import causal_backtest as cb
import hmm_guards
from _synth import two_regime_returns, prices_from_returns

WARMUP, REFIT = 60, 21


def _tape(seed=7):
    # 60 + 40 + 60 = 160 returns -> 161 price rows; grid hits k = 60,81,...,144
    r = two_regime_returns(n_bull=60, n_bear=40, n_bull2=60, seed=seed)
    return prices_from_returns(r, ticker="QQQ")


def _archive_like_the_engine(df):
    """Fit at every boundary the way the production engine does, recording
    params-or-None."""
    out = {}
    x = np.log1p(df["QQQ"].pct_change()).dropna().values
    for k in range(WARMUP, x.size, REFIT):
        try:
            model, _r, states, _d = cb.fit_hmm(df.iloc[:k + 1], "QQQ",
                                               n_restarts=3)
            ok, _ = hmm_guards.fit_is_sane(model, states)
        except Exception:
            ok = False
        out[k] = hmm_guards.params_to_dict(model) if ok else None
    return out


def test_replay_from_params_matches_faultsoft_bar_for_bar():
    """FULL-FRAME equality, every column, every bar -- not just
    the final value."""
    df = _tape()
    live = cb.causal_pbull_path_faultsoft(df, "QQQ", refit_every=REFIT,
                                          warmup=WARMUP, n_restarts=3)
    replay = cb.causal_pbull_path_from_params(df, "QQQ",
                                              _archive_like_the_engine(df),
                                              refit_every=REFIT, warmup=WARMUP)
    # Holds because the archive holds the very params the faultsoft run fitted
    # (same restart recipe) -- the production situation by construction.
    pd.testing.assert_frame_equal(live.reset_index(drop=True),
                                  replay.reset_index(drop=True))
    assert len(replay) > 2 * REFIT          # multi-boundary, not vacuous


def test_replay_reproduces_retention_when_a_boundary_failed():
    """A None boundary retains the PRIOR params -- the transient-fit
    divergence mode the replay seam exists to close. Boundary selection is DYNAMIC: early
    boundaries on a short warmup are often insane (verified: seed 7 yields
    None at k=60 and k=81), so pick the first two boundaries that actually
    fitted rather than assuming ks[0]/ks[1] did."""
    df = _tape()
    archive = _archive_like_the_engine(df)
    sane = [k for k in sorted(archive) if archive[k] is not None]
    assert len(sane) >= 2, "fixture must yield >=2 sane boundaries; enlarge tape"
    # ASSERT the premise the docstring states rather than trusting it: this test only supplies "first boundary is None" coverage while
    # the early fits really are rejected. If an hmmlearn/sklearn change makes
    # k=60 fit, the coverage vanishes SILENTLY and the test still passes.
    assert archive[min(archive)] is None, \
        "fixture no longer covers first-boundary-None: earliest fit is now sane"
    k_keep, k_fail = sane[0], sane[1]
    failed = dict(archive)
    failed[k_fail] = None                     # the SECOND sane fit "failed"
    held = cb.causal_pbull_path_from_params(df, "QQQ", failed,
                                            refit_every=REFIT, warmup=WARMUP)
    # Locate k_fail's segment BY DATE against the price frame -- replay row
    # indices do NOT start at warmup when early boundaries were insane
    # (seed 7's replay has rows only from the first sane boundary).
    seg_start = df["Date"].iloc[k_fail + 1]       # first bar of k_fail's segment
    seg_end = df["Date"].iloc[min(k_fail + REFIT, len(df) - 1)]
    seg = held[(pd.to_datetime(held["Date"]) >= pd.to_datetime(seg_start))
               & (pd.to_datetime(held["Date"]) <= pd.to_datetime(seg_end))]
    assert len(seg) > 0
    assert set(seg["refit_bar"]) == {k_keep}      # RETAINED k_keep params
    assert k_fail not in set(held["refit_bar"])
    # and a replay NEVER fits:
    assert "fit_hmm" not in inspect.getsource(cb.causal_pbull_path_from_params)


def test_underflow_bar_held_and_degraded_matches_faultsoft():
    """Pairing check: on a synthetic underflow the raw oracle RAISES while
    both fault-soft paths hold.

    The archive keeps SANE params at the earlier boundaries and swaps only the
    LAST sane boundary for underflow-tiny variances, so the frame produces real
    p_bull bars and THEN degrades. Tiny params at EVERY boundary would underflow
    at t=0 of the very first segment and degrade every row to p_bull 0.5, which
    would collapse the per-bar hold assertion below to `0.5 == 0.5` and prove
    nothing. Boundary selection is
    dynamic for the same reason as the retention test above: on this warmup,
    seed 7 yields no sane fit at k=60 or k=81."""
    df = _tape()
    tiny = {"means": [1e-6, 2e-6], "variances": [1e-18, 1e-18],
            "transmat": [[0.99, 0.01], [0.01, 0.99]], "startprob": [0.5, 0.5]}
    archive = _archive_like_the_engine(df)
    sane = [k for k in sorted(archive) if archive[k] is not None]
    assert len(sane) >= 2, "fixture must yield >=2 sane boundaries; enlarge tape"
    archive[sane[-1]] = tiny                  # underflow only the FINAL segment
    replay = cb.causal_pbull_path_from_params(df, "QQQ", archive,
                                              refit_every=REFIT, warmup=WARMUP)
    assert replay["degraded"].any()           # held bars flagged
    assert not replay["degraded"].all()       # ...and the hold branch is LIVE
    # held-bar VALUE semantics, PER BAR: every degraded bar holds
    # exactly the value of the nearest preceding non-degraded bar (0.5 when no
    # sane bar precedes it in the whole frame); never NaN, never fresh:
    hold, held_nondefault = 0.5, False
    for _, row in replay.iterrows():
        if row["degraded"]:
            assert row["p_bull"] == pytest.approx(hold)
            held_nondefault |= abs(hold - 0.5) > 1e-12
        else:
            hold = float(row["p_bull"])
    assert held_nondefault, \
        "hold check vacuous: no degraded bar held a non-default value"
    assert not replay["p_bull"].isna().any()
    with pytest.raises(Exception):
        cb.forward_filter_pbull_path(tiny, np.log1p(
            df["QQQ"].pct_change()).dropna().values)   # raw oracle raises


def test_no_sane_prior_produces_no_rows():
    df = _tape()
    out = cb.causal_pbull_path_from_params(
        df, "QQQ", {k: None for k in range(WARMUP, 160, REFIT)},
        refit_every=REFIT, warmup=WARMUP)
    assert len(out) == 0


def test_existing_functions_bytes_unchanged():
    """Additive-only guard: these are pinned digests of the existing
    functions' source, so the replay seam cannot silently change them. Any
    edit to one of these functions, even a docstring-only one, fails here and
    needs a deliberate re-pin after confirming the change is comment- or
    docstring-only."""
    for fn, want in [
        (cb.causal_pbull_path,
         "a4ede62cce3ea11baa501215652e182564a1bde45b976c44eb68feb7b3b3edc8"),
        (cb.causal_pbull_path_faultsoft,
         "c88809f3a915b659860e543cd7bf029dd458ea9a6d8ccbbc88b7a5b0beac4c60"),
        (cb._safe_forward_path,
         "696e621dcf6f936ab733e278c7423f606c117ed76fc81f8739af82f00d5e6956"),
        (cb.forward_filter_pbull_path,
         "8b615e0c08ef3ffe45c7ae00c46a5ed987ccdc6a331660357ec150bf215e65a1"),
    ]:
        got = hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
        assert got == want, f"{fn.__name__} source drifted"
