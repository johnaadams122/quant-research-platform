import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM

import hmm_guards
from _synth import two_regime_returns, prices_from_returns
from regime_signal import fit_hmm


def _fitted_model_and_returns(seed=7):
    r = two_regime_returns(seed=seed)
    X = r.reshape(-1, 1)
    m = GaussianHMM(n_components=2, covariance_type="full",
                    n_iter=200, random_state=42)
    m.fit(X)
    return m, r


def test_params_to_dict_relabels_bull_as_index_1():
    m, _ = _fitted_model_and_returns()
    params = hmm_guards.params_to_dict(m)
    assert params["means"][1] > params["means"][0]          # index 1 = bull
    assert len(params["means"]) == 2
    assert len(params["variances"]) == 2
    tm = np.asarray(params["transmat"])
    assert tm.shape == (2, 2)
    assert np.allclose(tm.sum(axis=1), 1.0, atol=1e-8)


def test_forward_filter_matches_hmmlearn_terminal_posterior():
    """THE gating check for the causal sweep machinery.

    hmmlearn's predict_proba returns SMOOTHED posteriors, but at the final
    observation beta_T = 1 so smoothed == filtered. If our pure-forward
    (causal, past-only) filter reproduces hmmlearn's terminal value to 1e-8,
    the terminal read provably used no future data.
    """
    m, r = _fitted_model_and_returns()
    bull_idx = int(np.argmax(m.means_.ravel()))
    expected = float(m.predict_proba(r.reshape(-1, 1))[-1, bull_idx])
    got = hmm_guards.forward_filter_pbull(hmm_guards.params_to_dict(m), r)
    assert abs(got - expected) < 1e-8


def test_forward_filter_rejects_impossible_data():
    params = {"means": [-0.001, 0.001], "variances": [1e-12, 1e-12],
              "transmat": [[0.99, 0.01], [0.01, 0.99]], "startprob": [0.5, 0.5]}
    with pytest.raises(ValueError):
        hmm_guards.forward_filter_pbull(params, np.array([50.0]))  # 50 log-ret: pdf underflows to 0


def test_params_roundtrip_via_json(tmp_path):
    m, r = _fitted_model_and_returns()
    params = hmm_guards.params_to_dict(m)
    fp = str(tmp_path / "params.json")
    hmm_guards.save_params(params, fp, asof="2026-07-13")
    loaded = hmm_guards.load_params(fp)
    assert loaded is not None
    p1 = hmm_guards.forward_filter_pbull(params, r)
    p2 = hmm_guards.forward_filter_pbull(loaded, r)
    assert abs(p1 - p2) < 1e-12


def test_load_params_missing_or_corrupt_returns_none(tmp_path):
    assert hmm_guards.load_params(str(tmp_path / "nope.json")) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="ascii")
    assert hmm_guards.load_params(str(bad)) is None


def test_load_params_rejects_schema_violations(tmp_path):
    """"None if missing/corrupt" means ANY corrupt payload -- structurally
    valid JSON with a broken params block must also fail soft, or the
    frozen-fallback path crashes/poisons downstream."""
    good = {"means": [-0.001, 0.001], "variances": [0.0001, 0.0009],
            "transmat": [[0.9, 0.1], [0.1, 0.9]], "startprob": [0.5, 0.5]}

    def write_payload(name, params):
        fp = tmp_path / name
        # json.dumps default allow_nan=True writes float('nan') as the literal
        # NaN token, which json.load reads back -- exactly the poisoned-but-
        # parseable file the guard must reject.
        fp.write_text(json.dumps({"schema": 1, "asof": "x", "params": params}),
                      encoding="ascii")
        return str(fp)

    # control: the well-formed payload loads fine
    assert hmm_guards.load_params(write_payload("good.json", good)) is not None

    cases = {
        "missing_startprob": {k: v for k, v in good.items() if k != "startprob"},
        "means_null": {**good, "means": None},
        "startprob_wrong_len": {**good, "startprob": [1.0]},
        "means_non_numeric": {**good, "means": ["a", "b"]},
        "transmat_nested_non_numeric": {**good,
                                        "transmat": [[0.9, 0.1], [0.1, "bad"]]},
        "variances_nan": {**good, "variances": [0.1, float("nan")]},
    }
    for name, params in cases.items():
        fp = write_payload(name + ".json", params)
        assert hmm_guards.load_params(fp) is None, f"{name} should return None"


def test_fit_is_sane_accepts_a_healthy_two_regime_fit():
    df = prices_from_returns(two_regime_returns())
    model, r, states, dates = fit_hmm(df, "PX", n_restarts=3)
    ok, reasons = hmm_guards.fit_is_sane(model, states)
    assert ok, f"healthy fit rejected: {reasons}"


def test_fit_is_sane_rejects_collapsed_and_degenerate_fits():
    m, _ = _fitted_model_and_returns()
    # (a) one state nearly unpopulated
    states_collapsed = np.zeros(500, dtype=int)
    states_collapsed[:3] = 1
    ok, reasons = hmm_guards.fit_is_sane(m, states_collapsed)
    assert not ok and "state_collapse" in reasons
    # (b) means indistinguishable
    m2, r2 = _fitted_model_and_returns()
    m2.means_ = np.array([[0.0004], [0.0004 + 1e-6]])
    states = m2.predict(r2.reshape(-1, 1))
    ok, reasons = hmm_guards.fit_is_sane(m2, states)
    assert not ok and "means_too_close" in reasons
    # (c) NaN-poisoned parameters: every threshold check is NaN-blind
    # (nan<x and nan>x are both False), so finiteness must short-circuit.
    m3, r3 = _fitted_model_and_returns()
    m3.means_ = np.array([[np.nan], [0.0008]])
    states3 = np.array([0, 1] * 250)
    ok, reasons = hmm_guards.fit_is_sane(m3, states3)
    assert not ok and reasons == ["nonfinite_params"]
