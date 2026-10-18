"""Tests for src/mining/batch5/ipo_long_run_underperformance_fade_beta_neutral.py.

Contract shell + dispatch-safety tests for the beta-neutral variant of
ipo_long_run_underperformance_fade. The account-curve maths is tested in
test_mining_beta_neutral_overlay.py; the bespoke driver in
test_run_ipo_fade_beta_neutral_abi1.py. This family is ALWAYS run through its
own driver (scripts/run_ipo_fade_beta_neutral_abi1.py), never the generic
scripts/run_abi1_mining.py dispatch.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC = os.path.join(_ROOT, "src")
_SCRIPTS = os.path.join(_ROOT, "scripts")
for _path in (_SRC, _SCRIPTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mining import beta_neutral_overlay as bno  # noqa: E402
from mining.batch5 import ipo_long_run_underperformance_fade_beta_neutral as mod  # noqa: E402

_CATALOG_PATH = os.path.join(_ROOT, "data", "mining", "catalog.json")
_FAMILY = "ipo_long_run_underperformance_fade_beta_neutral"


def _catalog_entry(family_id: str = _FAMILY) -> dict:
    with open(_CATALOG_PATH, "r", encoding="utf-8") as fh:
        catalog = json.load(fh)
    for entry in catalog:
        if entry.get("family_id") == family_id:
            return entry
    raise AssertionError(f"{family_id} not found in data/mining/catalog.json")


def _calendar(periods=60):
    return pd.bdate_range("2013-10-16", periods=periods)


def _frame(index, seed=3):
    rng = np.random.RandomState(seed)
    close = 20.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(index))))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.002,
                         "Low": np.minimum(open_, close) * 0.998,
                         "Close": close}, index=index)


# -------------------------------------------------------------- contract

def test_module_constants_match_the_candidate_contract():
    assert mod.FAMILY_ID == _FAMILY
    assert mod.SHAPE == "single_inverse_product"   # contract LABEL only, see docstring
    assert mod.TRADED == ["IPO"]
    assert mod.SIGNAL == []
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "SPY"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int) and mod.EXPECTED_MIN_TRADES > 0
    assert callable(mod.make_signal_fn)


def test_parameters_are_byte_identical_to_the_catalog_entry():
    entry = _catalog_entry()
    assert (json.dumps(mod.PARAMETERS, sort_keys=True)
            == json.dumps(entry["parameters"], sort_keys=True))
    assert set(mod.TRADED) | set(mod.SIGNAL) <= set(entry["tickers"])
    assert "SPY" in entry["tickers"]   # the hedge / benchmark leg


def test_parameters_are_tied_to_the_engine_constants_they_describe():
    assert mod.PARAMETERS["beta_window_sessions"] == bno.BETA_WINDOW_SESSIONS == 252
    assert mod.PARAMETERS["overlay_notional_fraction"] == bno.OVERLAY_NOTIONAL == 0.25
    assert mod.PARAMETERS["rebalance"] == "monthly_first_session"
    assert mod.PARAMETERS["hedge_leg"] == "SPY" == mod.PARAMETERS["benchmark"]


def test_catalog_entry_is_a_distinct_honest_new_family():
    entry = _catalog_entry()
    naked = _catalog_entry("ipo_long_run_underperformance_fade")
    assert entry["family_id"] != naked["family_id"]
    assert entry["status"] == "eligible"
    assert entry["account_types"] == ["taxable"]        # a short needs a margin account
    assert entry["added_date"] == "2026-09-23"
    assert "Ritter" in entry["source"] and "1991" in entry["source"]
    assert "beta" in entry["mechanism"].lower()
    for text in json.dumps(entry).split(" "):
        assert not _DOLLAR_FIGURE.search(text), text      # no dollar figures in catalog text


_DOLLAR_FIGURE = re.compile(r"\$\s?\d")


def test_the_dollar_figure_pattern_actually_matches_a_dollar_figure():
    # built at runtime so THIS file never contains a literal dollar amount
    # (the scan test below covers this file too)
    dollar = chr(36)
    assert _DOLLAR_FIGURE.search("costs " + dollar + "500 a year")
    assert _DOLLAR_FIGURE.search(dollar + "5")
    assert not _DOLLAR_FIGURE.search("no money here, 25% of equity")


def test_no_dollar_figure_literal_in_any_new_source_or_test_file():
    """No dollar-figure literal anywhere in this family's source or test
    files (docs, docstrings, comments and TESTS included): money figures are
    kept out of the code base, so catch one before it is committed."""
    for rel in ("src/mining/beta_neutral_overlay.py",
                "src/mining/batch5/ipo_long_run_underperformance_fade_beta_neutral.py",
                "scripts/run_ipo_fade_beta_neutral_abi1.py",
                "src/tests/test_mining_beta_neutral_overlay.py",
                "src/tests/test_batch5_ipo_long_run_underperformance_fade_beta_neutral.py",
                "src/tests/test_run_ipo_fade_beta_neutral_abi1.py"):
        with open(os.path.join(_ROOT, rel), "r", encoding="utf-8") as fh:
            text = fh.read()
        assert not _DOLLAR_FIGURE.search(text), rel


def test_driver_discovery_validates_this_module_contract_shape():
    import run_batch5_mining as driver
    spec = driver.validate_module_contract(mod, repo_root=_ROOT)
    assert spec["family_id"] == _FAMILY
    assert spec["shape"] == "single_inverse_product"
    assert spec["traded"] == ["IPO"]
    assert spec["benchmark"] == "SPY"
    assert spec["parameters"] == mod.PARAMETERS


def test_build_curve_uses_the_pinned_parameters():
    idx = _calendar(400)
    spy = _frame(idx, 2)
    ipo_close = 20.0 * np.cumprod(1.0 + 1.2 * spy["Close"].pct_change().fillna(0.0).to_numpy()
                                  + np.random.RandomState(1).normal(0.0, 0.006, len(idx)))
    ipo_open = np.concatenate([[ipo_close[0]], ipo_close[:-1]])
    ipo = pd.DataFrame({"Open": ipo_open, "High": np.maximum(ipo_open, ipo_close),
                        "Low": np.minimum(ipo_open, ipo_close), "Close": ipo_close},
                       index=idx)
    via_module = mod.build_curve(ipo, spy, ipo_cost_bps=9.0, spy_cost_bps=1.0,
                                 spy_entry_cost_bps=0.2)
    direct = bno.build_beta_neutral_overlay_curve(
        ipo, spy, ipo_cost_bps=9.0, spy_cost_bps=1.0, spy_entry_cost_bps=0.2,
        overlay_notional=mod.PARAMETERS["overlay_notional_fraction"],
        beta_window=mod.PARAMETERS["beta_window_sessions"])
    np.testing.assert_array_equal(via_module["strategy_daily"].to_numpy(),
                                  direct["strategy_daily"].to_numpy())
    pd.testing.assert_frame_equal(via_module["trades_df"], direct["trades_df"])


# ----------------------------------------------------------- dispatch safety

def test_generic_dispatch_call_shape_raises_instead_of_scoring_anything():
    """scripts/run_abi1_mining.py calls module.make_signal_fn(symbol, index,
    frames) with three positional args. This family has no per-bar signal --
    its account is a benchmark-plus-overlay curve only its own driver can
    build -- so that call shape must fail loudly, never return a signal that
    would score a plain long IPO position."""
    idx = _calendar(50)
    with pytest.raises(RuntimeError, match="bespoke driver"):
        mod.make_signal_fn("IPO", idx, {"IPO": _frame(idx)})


def test_make_signal_fn_raises_even_with_keyword_acknowledgements():
    idx = _calendar(50)
    with pytest.raises(RuntimeError, match="bespoke driver"):
        mod.make_signal_fn("IPO", idx, {"IPO": _frame(idx)},
                           driver_acknowledges_short_direction=True)


def test_real_generic_dispatch_builder_fails_closed_for_this_family():
    import run_abi1_mining as generic
    import run_batch5_mining as driver
    idx = _calendar(80)
    frames = {"IPO": _frame(idx, 1), "SPY": _frame(idx, 2)}
    plan = {"spec": driver.validate_module_contract(mod, repo_root=_ROOT),
            "module": mod, "effective": ["IPO"],
            "costs": {"IPO": {"cost_bps": 10.0}}}
    with pytest.raises(RuntimeError, match="bespoke driver"):
        generic._build_curve(plan, frames)


def test_family_is_not_in_any_accepted_scope_file():
    """No accepted.json may name this family: its only legitimate entry point
    is the bespoke driver. If one ever does, read the dispatch guard first.
    The scope files live under run_batch3_mining.RESEARCH_ROOT; on a checkout
    without that tree the glob finds nothing and this test checks nothing."""
    import run_batch3_mining as batch3
    pattern = os.path.join(batch3.RESEARCH_ROOT, "**", "accepted.json")
    for path in glob.glob(pattern, recursive=True):
        with open(path, "r", encoding="utf-8") as fh:
            assert _FAMILY not in fh.read(), path
