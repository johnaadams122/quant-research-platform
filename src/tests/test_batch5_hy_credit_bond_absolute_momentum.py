"""Tests for src/mining/batch5/hy_credit_bond_absolute_momentum.py.

This family wraps an existing signals factory in the batch5 module
contract -- the math is entirely delegated to
mining.signals.make_monthly_absolute_momentum_target_fn, so these tests
verify the WRAPPING (contract shape, parameter fidelity), not the underlying momentum math itself (that math already has
its own coverage via mining.signals's own tests).

The real-data end-to-end checks (live market-history fetches) are not part of
this public file; every test here is offline and uses synthetic data.
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _path in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mining import rotation_engine, signals  # noqa: E402
from mining.batch5 import hy_credit_bond_absolute_momentum as mod  # noqa: E402

FAMILY_ID = "hy_credit_bond_absolute_momentum"
CATALOG_PATH = os.path.join(_ROOT, "data", "mining", "catalog.json")
MODULE_PATH = os.path.join(_ROOT, "src", "mining", "batch5", f"{FAMILY_ID}.py")
RISK, DEFENSIVE = "HYG", "BIL"


def _catalog_entry() -> dict:
    with open(CATALOG_PATH, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    matches = [entry for entry in payload if entry.get("family_id") == FAMILY_ID]
    assert len(matches) == 1, (
        f"expected exactly one catalog.json entry for {FAMILY_ID!r}, "
        f"found {len(matches)}")
    return matches[0]


# --------------------------------------------------------------- contract

def test_parameters_are_byte_identical_to_the_catalog_json_entry():
    entry = _catalog_entry()
    assert (json.dumps(mod.PARAMETERS, sort_keys=True)
            == json.dumps(entry["parameters"], sort_keys=True))
    assert mod.FAMILY_ID == FAMILY_ID
    assert mod.SHAPE == "rotation_two_asset"
    assert mod.TRADED == [RISK, DEFENSIVE] == entry["tickers"]
    assert mod.SIGNAL == []
    assert mod.SNAPSHOTS == {}
    # ABI-1 always benchmarks SPY, deliberately NOT the catalog's own
    # "benchmark" concept (this family_id's catalog entry has no
    # benchmark_ticker field at all -- see the module's own docstring on
    # this hard pipeline rule).
    assert mod.BENCHMARK == "SPY"


def test_module_passes_the_driver_contract_in_isolation(tmp_path):
    """Discovery on a private copy of the module, so the result does not
    depend on whatever else sits in the real batch5 dir."""
    import run_batch5_mining as batch5

    modules_dir = tmp_path / "batch5"
    modules_dir.mkdir()
    shutil.copy(MODULE_PATH, modules_dir / f"{FAMILY_ID}.py")
    found, problems = batch5.discover_modules(str(modules_dir))
    assert problems == []
    assert list(found) == [FAMILY_ID]
    spec = found[FAMILY_ID]["spec"]
    assert spec["shape"] == "rotation_two_asset"
    assert spec["traded"] == [RISK, DEFENSIVE]
    assert spec["signal"] == []
    assert spec["benchmark"] == "SPY"
    catalog_params = _catalog_entry()["parameters"]
    assert batch5.parameter_drift(spec["parameters"], catalog_params) is None


def test_make_target_fn_requires_both_legs():
    lone = {RISK: pd.DataFrame({"Open": [1.0], "High": [1.0],
                                "Low": [1.0], "Close": [1.0]},
                               index=pd.bdate_range("2020-01-01", periods=1))}
    with pytest.raises(ValueError, match=DEFENSIVE):
        mod.make_target_fn(lone)


def test_make_target_fn_delegates_to_the_named_signals_factory():
    """Behavioral identity with signals.make_monthly_absolute_momentum_
    target_fn (RISK/DEFENSIVE/lookback_months=6) across many truncations --
    proves make_target_fn adds no extra decision logic of its own. The
    factory's rule looks only at RISK's OWN trailing return (DEFENSIVE's
    price never enters the decision, only the required-frame check), so
    RISK rises for the first two-thirds of the window and then declines
    hard for the rest -- a deterministic path guaranteed to pass through
    warm-up (None), a rising regime (RISK), and -- once the decline has
    run for over six calendar months -- a declining regime (DEFENSIVE)."""
    index = pd.bdate_range("2020-01-01", periods=440)
    up = np.arange(300)
    down = np.arange(140)
    risk_close = np.concatenate([
        100.0 * np.exp(0.0015 * up),
        100.0 * np.exp(0.0015 * up[-1]) * np.exp(-0.006 * down),
    ])
    def_close = np.full(len(index), 50.0)
    frames = {
        RISK: pd.DataFrame({"Open": risk_close, "High": risk_close * 1.001,
                            "Low": risk_close * 0.999, "Close": risk_close},
                           index=index),
        DEFENSIVE: pd.DataFrame({"Open": def_close, "High": def_close * 1.001,
                                 "Low": def_close * 0.999, "Close": def_close},
                                index=index),
    }
    ours = mod.make_target_fn(frames)
    reference = signals.make_monthly_absolute_momentum_target_fn(
        RISK, DEFENSIVE, lookback_months=mod._LOOKBACK_MONTHS)
    saw_risk = saw_defensive = saw_none = False
    for date in index:
        truncated = {key: frame.loc[frame.index <= date]
                    for key, frame in frames.items()}
        a, b = ours(truncated), reference(truncated)
        assert a == b, date
        saw_risk = saw_risk or a == RISK
        saw_defensive = saw_defensive or a == DEFENSIVE
        saw_none = saw_none or a is None
    assert saw_risk and saw_defensive and saw_none
