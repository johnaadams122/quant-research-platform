"""Tests for scripts/run_gold_copper_overlay_abi1.py -- the bespoke ABI-1 driver
for gold_copper_ratio_fear_valuation_spy_overlay.

Every run below uses SYNTHETIC price frames injected through the driver's
``ohlc_loader`` hook, a SYNTHETIC frozen futures cache written under tmp_path and
a TEMPORARY ledger; nothing here reads the network or touches
data/mining/trial_ledger.jsonl or the real frozen data/mining/gold_copper cache.
The dry-run tests prove wiring only -- never a real cost measurement (the quotes
file is synthetic on purpose). The family's catalog entry is read from
data/mining/catalog.json.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
SCRIPTS_ROOT = os.path.join(REPO_ROOT, "scripts")
for _path in (SRC_ROOT, SCRIPTS_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_gold_copper_overlay_abi1 as driver  # noqa: E402
from mining import abi1_driver, account_relative, constants, cost_tax_inputs, evidence, long_overlay as lo, trial_ledger  # noqa: E402
from mining.batch5 import gold_copper_ratio_fear_valuation_spy_overlay as mod  # noqa: E402
from mining.batch5 import treasury_month_end_index_demand_overlay as cal  # noqa: E402

_FAMILY = "gold_copper_ratio_fear_valuation_spy_overlay"
_ENTRY_PATH = os.path.join(REPO_ROOT, "data", "mining", "catalog.json")  # the family's shipped catalog entry


def _catalog_entry_from(fh):
    """This family's entry in the shipped catalog."""
    return next(entry for entry in json.load(fh) if entry["family_id"] == _FAMILY)


_DOLLAR_FIGURE = re.compile(r"\$\s?\d")
_EIGHT_DIGITS = re.compile(r"(?<![\d.])\b\d{8}\b(?![\d.])")  # word-bounded, so a hex digest is not a hit
_SPY_QUOTE = {"bid": 750.00, "ask": 750.15}       # about 2.0 bps
_REAL_FILES = [os.path.join(REPO_ROOT, "data", "mining", "trial_ledger.jsonl")] + [
    os.path.join(REPO_ROOT, "data", "mining", "gold_copper", name)
    for name in ("manifest.json", "yahoo_GC_F.csv", "yahoo_HG_F.csv")]


def _snapshot():
    return {p: ((os.stat(p).st_size, os.stat(p).st_mtime_ns) if os.path.exists(p) else None) for p in _REAL_FILES}


@pytest.fixture(scope="module", autouse=True)
def _the_real_ledger_and_the_real_frozen_cache_are_untouched_by_this_suite():
    """Every write in this file must land under tmp_path: the real trial ledger and
    the real frozen futures cache must be unchanged (size and mtime) when it ends."""
    before = _snapshot()
    yield
    assert _snapshot() == before


# ------------------------------------------------------------- fixtures

def _sessions(start, end):
    return cal.exchange_sessions(pd.Timestamp(start), pd.Timestamp(end))


def _walk(index, seed, vol, drift, start):
    r = np.random.RandomState(seed).normal(drift, vol, len(index))
    r[0] = 0.0
    return pd.Series(start * np.cumprod(1.0 + r), index=index)


def _level_path(n_months):
    level = np.full(n_months, 5.0)
    level[65:76] = 5.6
    level[76:86] = 4.9
    level[86:96] = 5.6
    level[96:] = 4.8
    return level


def _world(end="2013-12-31", seed=21, spy_start="2002-01-02", irx_close=None):
    """Production-shaped SYNTHETIC futures, SPY and ^IRX on the real exchange
    sessions (SPY lists long before the account starts, as in production)."""
    idx = _sessions("2000-08-30", end)
    rs = np.random.RandomState(seed)
    hg = _walk(idx, seed, 0.010, 0.0, 100.0)
    month_number = (idx.year - 2000) * 12 + idx.month - 8
    level = _level_path(int(month_number.max()) + 1)[month_number]
    gc = hg * np.exp(level + rs.normal(0.0, 0.01, len(idx)))
    spy_idx = _sessions(spy_start, end)
    spy_close = _walk(spy_idx, seed + 2, 0.010, 0.0004, 100.0).to_numpy()
    open_ = np.concatenate([[spy_close[0]], spy_close[:-1]])
    spy = pd.DataFrame({"Open": open_, "High": np.maximum(open_, spy_close) * 1.001,
                        "Low": np.minimum(open_, spy_close) * 0.999, "Close": spy_close}, index=spy_idx)
    irx_idx = _sessions("2000-01-03", end)
    values = np.linspace(1.0, 4.0, len(irx_idx)) if irx_close is None else irx_close(irx_idx)
    irx = pd.DataFrame({"Open": values, "High": values, "Low": values, "Close": values}, index=irx_idx)
    return {"gc": gc, "hg": hg, "SPY": spy, "^IRX": irx}


def _ohlcv(close):
    return pd.DataFrame({"Open": close.shift(1).fillna(close.iloc[0]), "High": close * 1.002,
                         "Low": close * 0.998, "Close": close, "Volume": 100}, index=close.index)


def _freeze(directory, world, *, through=None):
    def fetch(symbol):
        frame = _ohlcv(world["gc"] if symbol == "GC=F" else world["hg"])
        frame.index = pd.DatetimeIndex(frame.index).tz_localize("America/New_York")
        return frame
    end = through or world["gc"].index[-1].strftime("%Y-%m-%d")
    return mod.freeze_futures(str(directory), fetcher=fetch, through=end, retrieved_utc="2026-09-24T12:00:00Z")


def _loader(world):
    def load(symbol):
        frame = world[symbol].copy()
        frame.insert(0, "Date", frame.index)
        return frame.reset_index(drop=True)
    return load


def _quotes_payload(*, synthetic: bool, overrides=None) -> dict:
    payload = {
        "sampled_at_local": "2026-09-24T10:00:00",
        "market_state": "SYNTHETIC-test" if synthetic else "open",
        "source": "SYNTHETIC unit-test fixture" if synthetic else "unit test capture",
        "quotes": {"SPY": {**_SPY_QUOTE, "security_status": "Normal", "suspect": False}},
    }
    for key, value in (overrides or {}).items():         # a None value removes the key
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value
    return payload


def _fixture(tmp_path, *, synthetic=True, status="eligible", parameters=None, world=None, futures=True,
             quotes_overrides=None):
    world = world or _world()
    with open(_ENTRY_PATH, "r", encoding="utf-8") as fh:
        entry = _catalog_entry_from(fh)
    entry["status"] = status
    if parameters is not None:
        entry["parameters"] = parameters
    catalog_path = str(tmp_path / "catalog.json")
    with open(catalog_path, "w", encoding="utf-8") as fh:
        json.dump([entry], fh)
    quotes_path = str(tmp_path / "quotes.json")
    with open(quotes_path, "w", encoding="utf-8") as fh:
        json.dump(_quotes_payload(synthetic=synthetic, overrides=quotes_overrides), fh)
    futures_dir = str(tmp_path / "futures")
    if futures:
        _freeze(futures_dir, world)
    return catalog_path, quotes_path, str(tmp_path / "ledger.jsonl"), str(tmp_path / "research"), futures_dir


def _call(tmp_path, *, dry_run=True, synthetic=True, world=None, run_id="gcr-test", n_random_draws=6,
          accept_open_final_window=False, futures_dir=None, futures=True, tested_date="2026-09-24",
          **fixture_kwargs):
    world = world or _world()
    catalog_path, quotes_path, ledger_path, research_root, default_futures = _fixture(
        tmp_path, synthetic=synthetic, world=world, futures=futures, **fixture_kwargs)
    result = driver.compute_and_record(
        catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path,
        run_id=run_id, tested_date=tested_date, dry_run=dry_run,
        signal_source_path="scripts/run_gold_copper_overlay_abi1.py",
        signal_source_sha256="a" * 64, research_root=research_root,
        ohlc_loader=_loader(world), futures_dir=futures_dir or default_futures,
        n_random_draws=n_random_draws, accept_open_final_window=accept_open_final_window)
    return result, ledger_path


def _mk(tmp_path, name):
    path = tmp_path / name
    path.mkdir()
    return path


def _cut(world, last):
    """Every series of the world cut at ``last`` (production shape: all end together)."""
    last = pd.Timestamp(last)
    return {"gc": world["gc"].loc[:last], "hg": world["hg"].loc[:last], "SPY": world["SPY"].loc[:last],
            "^IRX": world["^IRX"].loc[:last]}


def _prepared(world):
    return mod.prepare(world["SPY"], world["^IRX"], world["gc"], world["hg"])


def _last_high_start(ctx):
    states = [bool(s) for s in ctx["states"]]
    return max(k for k in range(1, len(states)) if states[k] and not states[k - 1])


_SCORED = ("survived", "cut_dsr", "cut_holdout", "cut_risk_refusal")


def _low_end_world():
    """The default world cut at its last month-end whose state is LOW, so the data
    ends with no window open and none entered on the last bar (the real-run
    tests must not depend on where the synthetic noise leaves the last state)."""
    world = _world()
    ctx = _prepared(world)
    k = max(i for i, state in enumerate(ctx["states"]) if not state)
    return _cut(world, ctx["decisions"][k])


@pytest.fixture(autouse=True)
def _small_pre_declared_control_configuration_and_no_real_default_path(monkeypatch, request, tmp_path):
    """Two safeguards for every test.

    1. A real run must use the pre-declared 500 draws; scoring 500 draws in every
       real-run test would take about a minute each, so the driver's own constant
       is shrunk for the suite.
    2. The driver's REAL default paths (ledger, catalog, research root, frozen
       futures cache) are redirected under tmp_path, so a test that forgets to
       pass a path, or a guard that stops holding (which could otherwise write
       a results file under the real research root), can never touch a real
       file.

    The tests that pin the real defaults carry ``pre_declared_default`` in their
    name and keep both."""
    if "pre_declared_default" in request.node.name:
        return
    monkeypatch.setattr(driver, "DEFAULT_RANDOM_DRAWS", 6)
    for name, leaf in (("LEDGER_PATH", "_guard_ledger.jsonl"), ("CATALOG_PATH", "_guard_catalog.json"),
                       ("RESEARCH_ROOT", "_guard_research"), ("FUTURES_DIR", "_guard_futures")):
        monkeypatch.setattr(driver, name, str(tmp_path / leaf))


# ------------------------------------------------------------ dry-run wiring

def test_dry_run_evaluates_end_to_end_without_writing_a_ledger_row(tmp_path):
    result, ledger_path = _call(tmp_path)
    assert result["family_id"] == _FAMILY
    assert "reason_code" not in result
    assert result["verdict"] in ("not_scoreable", "cut_dsr", "cut_risk_refusal",
                                 "holdout_not_evaluable", "cut_holdout", "survived")
    assert result["tax_policy"] == "realization" and result["benchmark"] == "SPY"
    assert result["tickers"] == ["SPY"]
    assert not os.path.exists(ledger_path)


def test_dry_run_reports_the_structural_details(tmp_path):
    result, _ = _call(tmp_path)
    diag = result["overlay_diagnostics"]
    assert diag["n_windows"] == result["n_trades"] >= 3
    assert diag["worst_window_net_ret"] < diag["best_window_net_ret"]
    assert diag["overlay_notional_fraction"] == 0.25
    assert diag["n_decisions"] > 60 and 0 < diag["n_high"] < diag["n_decisions"]
    assert diag["episode_months"] and max(diag["episode_months"]) >= 5
    assert diag["first_decision"] == "2005-07-29"
    assert 0.0 < diag["overlay_session_share"] < 1.0
    assert result["time_in_market"] == 1.0                      # the SPY core is always on
    json.dumps(result, allow_nan=False, default=str)             # nothing non-finite leaks


def test_the_account_lives_on_spys_sessions_from_the_first_decision_month_end(tmp_path):
    world = _world()
    result, _ = _call(tmp_path, world=world)
    assert result["account_start"] == "2005-07-29"
    assert result["account_end"] == world["SPY"].index[-1].strftime("%Y-%m-%d")
    assert world["SPY"].index[0] < pd.Timestamp("2005-07-29")


def test_the_overlay_share_of_equity_is_reported_and_matches_an_independent_recomputation(tmp_path):
    world = _world()
    result, _ = _call(tmp_path, world=world)
    diag = result["overlay_diagnostics"]
    ctx = _prepared(world)
    curve = mod.build_curve(world["SPY"], world["^IRX"], world["gc"], world["hg"], context=ctx,
                            spy_cost_bps=cost_tax_inputs.compute_spread_bps(_SPY_QUOTE["bid"], _SPY_QUOTE["ask"]),
                            spy_entry_cost_bps=cost_tax_inputs.compute_spread_bps(_SPY_QUOTE["bid"], _SPY_QUOTE["ask"]) / 2)
    idx = curve["strategy_daily"].index
    equity = np.cumprod(1.0 + curve["strategy_daily"].to_numpy())
    close = world["SPY"]["Close"].loc[idx].to_numpy()
    shares_of_equity = []
    for _, trade in curve["trades_df"].iterrows():
        a, b = idx.get_loc(trade["entry_date"]), idx.get_loc(trade["exit_date"])
        o = 0.25 * equity[a] / close[a]
        shares_of_equity.extend(o * close[a + 1:b + 1] / equity[a + 1:b + 1])
    assert diag["overlay_share_of_equity_min"] == pytest.approx(min(shares_of_equity), rel=1e-9)
    assert diag["overlay_share_of_equity_max"] == pytest.approx(max(shares_of_equity), rel=1e-9)
    assert diag["overlay_share_of_equity_min"] < 0.25 < diag["overlay_share_of_equity_max"] or \
        diag["overlay_share_of_equity_min"] <= 0.25 <= diag["overlay_share_of_equity_max"]


def test_the_cost_comes_from_the_spy_quote_the_leg_at_the_full_spread_the_benchmark_at_half(tmp_path):
    result, _ = _call(tmp_path)
    diag = result["overlay_diagnostics"]
    spread = cost_tax_inputs.compute_spread_bps(_SPY_QUOTE["bid"], _SPY_QUOTE["ask"])
    assert diag["spy_leg_cost_bps"] == pytest.approx(spread)
    assert diag["benchmark_entry_cost_bps"] == pytest.approx(spread / 2.0)
    assert diag["spy_leg_cost_bps"] != diag["benchmark_entry_cost_bps"]


def test_the_financing_rate_is_the_bill_series_not_a_constant(tmp_path):
    world = _world(irx_close=lambda idx: np.full(len(idx), 4.0))
    result, _ = _call(tmp_path, world=world)
    expected = float(account_relative.coupon_equivalent_yield(0.04))
    assert result["overlay_diagnostics"]["mean_financing_rate_annual_on_overlay_sessions"] == pytest.approx(
        expected, rel=1e-12)


def test_a_negative_bill_print_is_loaded_without_the_positivity_gate(tmp_path):
    def with_negative_prints(idx):
        irx = np.full(len(idx), 1.0)
        irx[(idx >= "2009-03-19") & (idx <= "2009-03-27")] = -0.105
        return irx
    world = _world(irx_close=with_negative_prints)
    assert (world["^IRX"]["Close"] < 0).any()
    result, _ = _call(tmp_path, world=world)
    assert "reason_code" not in result and np.isfinite(result["sharpe_daily_excess"])


def test_a_traded_spy_with_a_non_positive_price_still_fails_closed(tmp_path):
    world = _world()
    world["SPY"].iloc[900, world["SPY"].columns.get_loc("Close")] = -1.0
    with pytest.raises(ValueError, match="SPY"):
        _call(tmp_path, world=world)


def test_the_spy_frame_really_drives_the_core_and_the_benchmark(tmp_path):
    a, _ = _call(_mk(tmp_path, "a"), world=_world())
    other = _world()
    other["SPY"] = _world(seed=99)["SPY"]
    b, _ = _call(_mk(tmp_path, "b"), world=other)
    assert a["sharpe_daily_excess"] != b["sharpe_daily_excess"]


def test_the_futures_really_drive_the_signal_and_so_the_curve(tmp_path):
    a, _ = _call(_mk(tmp_path, "a"), world=_world())
    other = _world()
    flat = other["gc"] * 0 + np.exp(np.random.RandomState(5).normal(0, 0.15, len(other["gc"]))) * 100.0
    other["gc"] = pd.Series(flat.to_numpy(), index=other["gc"].index)
    b, _ = _call(_mk(tmp_path, "b"), world=other)
    assert a["overlay_diagnostics"]["episode_months"] != b["overlay_diagnostics"]["episode_months"]


def test_the_scored_result_is_the_taxed_realization_evaluation_of_the_main_curve(tmp_path):
    """The recorded verdict must come from the realization-tax evaluation of the
    MERGED-WINDOW curve, independently rebuilt here from the module."""
    world = _world()
    result, _ = _call(tmp_path, world=world)
    spread = cost_tax_inputs.compute_spread_bps(_SPY_QUOTE["bid"], _SPY_QUOTE["ask"])
    curve = mod.build_curve(world["SPY"], world["^IRX"], world["gc"], world["hg"],
                            spy_cost_bps=spread, spy_entry_cost_bps=spread / 2.0)
    kwargs = dict(benchmark_frame=world["SPY"], benchmark_entry_cost_bps=spread / 2.0, account_type="taxable")
    taxed = abi1_driver.evaluate_family(curve, tax_policy="realization", **kwargs)
    untaxed = abi1_driver.evaluate_family(curve, tax_policy="none", **kwargs)
    assert taxed["layer1"]["sharpe_daily_excess"] != pytest.approx(untaxed["layer1"]["sharpe_daily_excess"], rel=1e-9)
    assert result["sharpe_daily_excess"] == pytest.approx(taxed["layer1"]["sharpe_daily_excess"], rel=1e-12)
    assert result["dsr_value"] == pytest.approx(taxed["layer1"]["dsr_value"], rel=1e-9)
    assert result["verdict"] == taxed["verdict"]
    assert result["overlay_diagnostics"]["sharpe_daily_excess_untaxed"] == pytest.approx(
        untaxed["layer1"]["sharpe_daily_excess"], rel=1e-12)


# -------------------------------------------------------------- risk layer

def test_dry_run_reports_the_four_risk_refusals_on_the_overlay_only_excess_account(tmp_path):
    result, _ = _call(tmp_path)
    risk = result["overlay_diagnostics"]["risk_layer"]
    assert risk["account_basis"] == "overlay_only_excess"           # overlay-only excess account basis
    assert risk["absolute_cap"] == -0.35
    for key in ("absolute_passed", "relative_passed", "red_week_streak_passed", "worst_stretch_passed",
                "all_refusals_passed"):
        assert isinstance(risk[key], bool), key
    assert isinstance(risk["failed_refusals"], list) and risk["worst_stretch_window_weeks"] == 8
    assert risk["whole_account_reference"]["gates_verdict"] is False


def test_the_first_dry_run_does_not_false_trip_the_absolute_cap_on_the_overlay_only_excess(tmp_path):
    """The overlay is a fraction of equity, so on the synthetic world the
    overlay-only drawdown stays well inside SPY's own (under 60 percent of it);
    the absolute cap must read the excess account, not the whole SPY-core account."""
    result, _ = _call(tmp_path)
    risk = result["overlay_diagnostics"]["risk_layer"]
    assert risk["absolute_passed"] is True
    assert risk["account_max_drawdown"] > risk["absolute_cap"]
    assert risk["account_max_drawdown"] > 0.6 * risk["benchmark_max_drawdown"]


def test_the_excess_is_exactly_zero_outside_the_windows_through_the_real_tax_path():
    world = _world()
    ctx = _prepared(world)
    curve = mod.curve_from_states(ctx, world["SPY"], ctx["states"], spy_cost_bps=2.0, spy_entry_cost_bps=1.0)
    result = abi1_driver.evaluate_family(curve, tax_policy="realization", benchmark_frame=world["SPY"],
                                         benchmark_entry_cost_bps=1.0, account_type="taxable")
    from mining import benchmark_relative
    bench = benchmark_relative.cost_adjusted_benchmark_daily_returns(world["SPY"], entry_cost_bps=1.0)
    excess = result["account_daily"] - bench.reindex(result["account_daily"].index)
    outside = ~curve["overlay_mask"].to_numpy()
    assert outside.sum() > 500 and (excess[outside] == 0.0).all()
    assert (excess[curve["overlay_mask"].to_numpy()] != 0.0).any()


def test_a_core_that_differs_from_the_benchmark_outside_the_windows_is_refused(tmp_path, monkeypatch):
    real = driver.mod.build_curve

    def drifted(*args, **kwargs):
        curve = dict(real(*args, **kwargs))
        daily = curve["strategy_daily"].copy()
        off = np.flatnonzero(~curve["overlay_mask"].to_numpy())
        daily.iloc[off[400]] += 1e-4
        curve["strategy_daily"] = daily
        return curve
    monkeypatch.setattr(driver.mod, "build_curve", drifted)
    with pytest.raises(RuntimeError, match="not overlay-only"):
        _call(tmp_path, dry_run=False, synthetic=False)
    assert not os.path.exists(str(tmp_path / "ledger.jsonl"))


def test_a_curve_that_lost_its_account_kind_is_refused_before_anything_is_recorded(tmp_path, monkeypatch):
    real = driver.mod.build_curve

    def stripped(*args, **kwargs):
        curve = dict(real(*args, **kwargs))
        curve.pop("account_kind")
        return curve
    monkeypatch.setattr(driver.mod, "build_curve", stripped)
    with pytest.raises(RuntimeError, match="overlay_only_excess"):
        _call(tmp_path, dry_run=False, synthetic=False)
    assert not os.path.exists(str(tmp_path / "ledger.jsonl"))


# ---------------------------------------------------------- the control arms

def _independent_arm_sharpes(world, curve_fn):
    spread = cost_tax_inputs.compute_spread_bps(_SPY_QUOTE["bid"], _SPY_QUOTE["ask"])
    ctx = _prepared(world)
    curve = curve_fn(ctx, spread)
    kwargs = dict(benchmark_frame=world["SPY"], benchmark_entry_cost_bps=spread / 2.0, account_type="taxable")
    taxed = abi1_driver.evaluate_family(curve, tax_policy="realization", **kwargs)
    untaxed = abi1_driver.evaluate_family(curve, tax_policy="none", **kwargs)
    return taxed, untaxed, curve


def test_the_always_on_arm_is_reported_and_equals_an_independent_evaluation(tmp_path):
    world = _world()
    result, _ = _call(tmp_path, world=world)
    arm = result["controls"]["always_on"]
    taxed, untaxed, curve = _independent_arm_sharpes(
        world, lambda ctx, spread: mod.build_always_on_curve(
            ctx, world["SPY"], spy_cost_bps=spread, spy_entry_cost_bps=spread / 2.0))
    assert arm["n_trades"] == len(curve["trades_df"]) == 1
    assert arm["sharpe_daily_excess"] == pytest.approx(taxed["layer1"]["sharpe_daily_excess"], rel=1e-12)
    assert arm["sharpe_daily_excess_untaxed"] == pytest.approx(untaxed["layer1"]["sharpe_daily_excess"], rel=1e-12)
    assert arm["dsr_value"] == pytest.approx(taxed["layer1"]["dsr_value"], rel=1e-9)
    assert arm["verdict"] == taxed["verdict"]
    assert arm["sharpe_daily_excess"] != result["sharpe_daily_excess"]      # a genuinely different arm
    assert arm["overlay_session_share"] > 0.99


def test_the_random_window_arm_reports_the_draws_and_equals_an_independent_evaluation(tmp_path):
    world = _world()
    result, _ = _call(tmp_path, world=world, n_random_draws=5)
    arm = result["controls"]["random_window"]
    assert arm["n_draws"] == 5 and arm["seed"] == "2026_0924"
    assert int(arm["seed"].replace("_", "")) == mod.RANDOM_CONTROL_SEED     # a string: no eight-digit integer in a record
    spread = cost_tax_inputs.compute_spread_bps(_SPY_QUOTE["bid"], _SPY_QUOTE["ask"])
    ctx = _prepared(world)
    draws = mod.random_window_state_draws(ctx["states"], 5, seed=mod.RANDOM_CONTROL_SEED)
    expected, expected_untaxed = [], []
    kwargs = dict(benchmark_frame=world["SPY"], benchmark_entry_cost_bps=spread / 2.0, account_type="taxable")
    for draw in draws:
        curve = mod.curve_from_states(ctx, world["SPY"], draw, spy_cost_bps=spread, spy_entry_cost_bps=spread / 2.0)
        expected.append(abi1_driver.evaluate_family(curve, tax_policy="realization", **kwargs)["layer1"]["sharpe_daily_excess"])
        expected_untaxed.append(abi1_driver.evaluate_family(curve, tax_policy="none", **kwargs)["layer1"]["sharpe_daily_excess"])
    assert arm["draw_sharpes"] == pytest.approx(expected, rel=1e-12)
    assert arm["draw_sharpes_untaxed"] == pytest.approx(expected_untaxed, rel=1e-12)
    main = result["sharpe_daily_excess"]
    below = sum(v < main for v in expected) + 0.5 * sum(v == main for v in expected)
    assert arm["main_percentile"] == pytest.approx(below / 5)
    assert arm["draws_at_or_above_main"] == sum(v >= main for v in expected)
    assert len({round(v, 12) for v in arm["draw_sharpes"]}) > 1                # the draws genuinely differ


def test_the_random_window_arm_keeps_the_main_arms_episode_structure(tmp_path):
    result, _ = _call(tmp_path, n_random_draws=4)
    arm = result["controls"]["random_window"]
    diag = result["overlay_diagnostics"]
    assert arm["episode_months"] == diag["episode_months"]
    assert arm["high_share"] == pytest.approx(diag["n_high"] / diag["n_decisions"])


def test_the_control_arms_are_deterministic_across_runs(tmp_path):
    a, _ = _call(_mk(tmp_path, "a"), n_random_draws=4)
    b, _ = _call(_mk(tmp_path, "b"), n_random_draws=4)
    assert a["controls"] == b["controls"]


@pytest.mark.parametrize("main,always,pct,expected", [
    (0.10, 0.05, 0.96, True), (0.10, 0.05, 0.95, True), (0.10, 0.05, 0.949, False),
    (0.10, 0.12, 0.99, False), (0.10, 0.10, 0.99, False),
    (None, 0.05, 0.99, None), (0.10, None, 0.99, None), (0.10, 0.05, None, None)])
def test_the_pre_declared_interpretation_rule(main, always, pct, expected):
    verdict = driver.interpret_controls(main_sharpe=main, always_on_sharpe=always, main_percentile=pct)
    assert verdict["signal_evidence"] is expected
    assert verdict["percentile_bar"] == 0.95
    assert "always-on" in verdict["rule"] and "95th percentile" in verdict["rule"]


def test_the_percentile_is_a_mid_rank_over_the_scored_draws_only():
    assert driver._percentile(0.1, [0.1, 0.2, 0.0, None]) == (pytest.approx(0.5), 2)     # one below, one tie, one above
    assert driver._percentile(0.3, [0.1, 0.2, 0.0]) == (1.0, 0)
    assert driver._percentile(-1.0, [0.1, 0.2]) == (0.0, 2)
    assert driver._percentile(None, [0.1]) == (None, None) and driver._percentile(0.1, [None]) == (None, None)


def test_the_controls_block_states_the_interpretation_for_this_run(tmp_path):
    result, _ = _call(tmp_path, n_random_draws=4)
    controls = result["controls"]
    main = result["sharpe_daily_excess"]
    expected = driver.interpret_controls(main_sharpe=main, always_on_sharpe=controls["always_on"]["sharpe_daily_excess"],
                                         main_percentile=controls["random_window"]["main_percentile"])
    assert controls["interpretation"] == expected
    assert controls["ledger_rows_written_for_controls"] == 0


def test_a_real_run_refuses_a_non_default_random_draw_count_and_writes_nothing(tmp_path):
    result, ledger_path = _call(tmp_path, dry_run=False, synthetic=False, n_random_draws=7)
    assert result["reason_code"] == "control_configuration_drift"
    assert "verdict" not in result and not os.path.exists(ledger_path)


def test_pre_declared_default_a_dry_run_may_use_fewer_draws(tmp_path):
    assert driver.DEFAULT_RANDOM_DRAWS == mod.RANDOM_CONTROL_DRAWS >= 200
    result, _ = _call(tmp_path, n_random_draws=3)
    assert result["controls"]["random_window"]["n_draws"] == 3


def test_a_failing_control_arm_aborts_the_real_run_before_anything_is_recorded(tmp_path, monkeypatch):
    """The ledger note refers to the controls, so a permanent row must never be
    recorded without them: a control failure propagates (fail closed)."""
    def boom(*args, **kwargs):
        raise RuntimeError("control arm failed")
    monkeypatch.setattr(driver.mod, "build_always_on_curve", boom)
    with pytest.raises(RuntimeError, match="control arm failed"):
        _call(tmp_path, dry_run=False, synthetic=False, world=_low_end_world())
    assert not os.path.exists(str(tmp_path / "ledger.jsonl"))


def test_the_control_arms_write_no_ledger_row_of_their_own(tmp_path):
    result, ledger_path = _call(tmp_path, dry_run=False, synthetic=False)
    rows = trial_ledger.read_all(ledger_path)
    assert sorted(r.get("event") for r in rows) == ["first_test", "outcome"]
    assert {r["family_id"] for r in rows} == {_FAMILY}


# ---------- the controls gate the PRE-REGISTERED pass, not the recorded stage
#
# The stage vocabulary is closed in shared code and every tested family is recorded, so a main ABI-1
# passing stage is still written to the ledger. What the driver adds is (a) a top-level pre_registered_pass
# (passing stage AND signal_evidence True) and (b) an unmissable first sentence of the driver-supplied note
# (and of the printed summary) whenever the stage passed but the controls did not. The synthetic world
# cannot reach the passing stage on its own, so the real-path tests below FORCE the verdict of the main scored
# evaluation only (the first evaluate_family call) and force the control condition through
# interpret_controls; the decision function and the clause builder are also tested directly.

_FAILED_FLAG = ("CONTROL CONDITION FAILED: the ABI-1 stage survived is NOT a survivor under this family's own "
                "pre-registered rule (beat the always-on control AND the 95th percentile of the random-window "
                "control). This is not a pass; do not promote it without review.")
_MET_CLAUSE = ("PRE-REGISTERED RULE MET: the ABI-1 stage survived AND the control condition holds (beat the "
               "always-on control AND the 95th percentile of the random-window control); "
               "pre_registered_pass=True.")
_NOT_SURVIVED = ("cut_dsr", "cut_holdout", "cut_risk_refusal", "not_scoreable", "holdout_not_evaluable")


@pytest.mark.parametrize("verdict,evidence,expected", (
    [("survived", True, True), ("survived", False, False), ("survived", None, False)]
    + [(v, True, False) for v in _NOT_SURVIVED] + [(v, False, False) for v in _NOT_SURVIVED]))
def test_the_pre_registered_pass_needs_both_the_abi1_survival_and_the_control_condition(verdict, evidence, expected):
    assert driver.pre_registered_pass(verdict, evidence) is expected


@pytest.mark.parametrize("evidence", [False, None])
def test_a_survived_stage_without_the_control_condition_carries_the_exact_unmissable_flag(evidence):
    assert driver.pre_registered_clause("survived", evidence) == _FAILED_FLAG


def test_a_survived_stage_whose_controls_hold_says_the_pre_registered_rule_was_met():
    clause = driver.pre_registered_clause("survived", True)
    assert clause == _MET_CLAUSE and "FAILED" not in clause


@pytest.mark.parametrize("verdict", _NOT_SURVIVED)
@pytest.mark.parametrize("evidence", [True, False, None])
def test_a_stage_that_did_not_survive_carries_no_clause(verdict, evidence):
    assert driver.pre_registered_clause(verdict, evidence) == ""


def _force_main_verdict_survived(monkeypatch):
    """The synthetic world cannot reach the passing stage: relabel the verdict of the MAIN scored evaluation
    only (compute_and_record's first evaluate_family call; every control arm keeps its real verdict)."""
    real = driver.abi1_driver.evaluate_family
    calls = {"n": 0}

    def wrapped(*args, **kwargs):
        result = real(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 1:
            result = dict(result)
            result["verdict"] = "survived"
        return result
    monkeypatch.setattr(driver.abi1_driver, "evaluate_family", wrapped)


def _force_signal_evidence(monkeypatch, value):
    real = driver.interpret_controls

    def forced(**kwargs):
        verdict = dict(real(**kwargs))
        verdict["signal_evidence"] = value
        return verdict
    monkeypatch.setattr(driver, "interpret_controls", forced)


_RECORDED_PREFIX = "cash_convention=sweep_cash_daily; benchmark=SPY; tax_policy=realization; "


@pytest.mark.parametrize("evidence", [False, None])
def test_a_recorded_survived_stage_that_failed_the_controls_is_flagged_first_and_is_not_a_pre_registered_pass(
        tmp_path, monkeypatch, evidence):
    _force_main_verdict_survived(monkeypatch)
    _force_signal_evidence(monkeypatch, evidence)
    result, ledger_path = _call(tmp_path, dry_run=False, synthetic=False, world=_low_end_world())
    assert result["verdict"] == "survived" and result["stage"] == "survived"        # recorded unconditionally
    assert result["pre_registered_pass"] is False
    assert result["pre_registered_clause"] == _FAILED_FLAG
    outcomes = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    assert len(outcomes) == 1 and outcomes[0]["stage"] == "survived"
    # the shared writer puts its fixed metadata prefix first; the driver's own note starts with the flag
    assert outcomes[0]["note"].startswith(_RECORDED_PREFIX + _FAILED_FLAG + " cost model:")
    assert outcomes[0]["note"].count("CONTROL CONDITION FAILED") == 1
    assert "pre_registered_pass=false" in outcomes[0]["note"].lower()


def test_a_recorded_survived_stage_that_passed_the_controls_says_the_rule_was_met(tmp_path, monkeypatch):
    _force_main_verdict_survived(monkeypatch)
    _force_signal_evidence(monkeypatch, True)
    result, ledger_path = _call(tmp_path, dry_run=False, synthetic=False, world=_low_end_world())
    assert result["stage"] == "survived" and result["pre_registered_pass"] is True
    assert result["pre_registered_clause"] == _MET_CLAUSE
    outcomes = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    assert outcomes[0]["note"].startswith(_RECORDED_PREFIX + _MET_CLAUSE + " cost model:")
    assert "CONTROL CONDITION FAILED" not in outcomes[0]["note"]
    assert "pre_registered_pass=true" in outcomes[0]["note"].lower()


def test_a_stage_that_did_not_survive_is_not_a_pre_registered_pass_whatever_the_controls_say(tmp_path, monkeypatch):
    _force_signal_evidence(monkeypatch, True)
    result, ledger_path = _call(tmp_path, dry_run=False, synthetic=False, world=_low_end_world())
    assert result["verdict"] != "survived" and result["pre_registered_pass"] is False
    assert result["pre_registered_clause"] == ""
    outcomes = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    assert outcomes[0]["note"].startswith(_RECORDED_PREFIX + "cost model:")     # no clause: unchanged prefix
    assert "CONTROL CONDITION FAILED" not in outcomes[0]["note"] and "PRE-REGISTERED RULE MET" not in outcomes[0]["note"]
    assert "pre_registered_pass=false" in outcomes[0]["note"].lower()


def test_the_printed_summary_says_the_same_as_the_note_when_the_controls_failed(tmp_path, monkeypatch, capsys):
    _force_main_verdict_survived(monkeypatch)
    _force_signal_evidence(monkeypatch, False)
    world = _low_end_world()
    catalog_path, quotes_path, ledger_path, research_root, futures_dir = _fixture(
        tmp_path, synthetic=False, world=world)
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(world))
    code = driver.main(["--quotes", quotes_path, "--catalog", catalog_path, "--ledger", ledger_path,
                        "--research-root", research_root, "--futures-dir", futures_dir,
                        "--run-id", "cli-real", "--tested-date", "2026-09-24"])
    assert code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert _FAILED_FLAG in lines[:-1]                        # its own printed line, before the JSON line
    printed = json.loads(lines[-1])
    assert printed["result"]["pre_registered_pass"] is False and printed["result"]["stage"] == "survived"


def test_the_printed_summary_stays_quiet_when_nothing_survived(tmp_path, monkeypatch, capsys):
    world = _low_end_world()
    catalog_path, quotes_path, ledger_path, research_root, futures_dir = _fixture(tmp_path, world=world)
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(world))
    assert driver.main(["--dry-run", "--quotes", quotes_path, "--catalog", catalog_path, "--ledger", ledger_path,
                        "--research-root", research_root, "--futures-dir", futures_dir, "--random-draws", "3"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["result"]["pre_registered_pass"] is False


# ---------- a REAL run needs a market-hours quote capture dated the run date
#
# refuse_synthetic_quotes (shared) only catches the SYNTHETIC marker; a closed-market or old capture
# passes it and would produce unusable cost samples. The driver also refuses, for real runs
# only, unless the capture is (i) market_state starting with open (any case), (ii) a parseable
# sampled_at_local (naive = US Eastern), (iii) dated the run's tested date, and (iv) taken between
# 09:45 and 15:45 Eastern inclusive. Reason code quotes_not_market_hours; nothing is written.

def _hours_refusal(overrides=None, tested="2026-09-24"):
    payload = {"sampled_at_local": "2026-09-24T10:00:00", "market_state": "open", "quotes": {}}
    for key, value in (overrides or {}).items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value
    return driver.quotes_market_hours_refusal(payload, tested)


def test_an_open_in_window_capture_dated_the_run_date_is_accepted():
    assert _hours_refusal() is None


@pytest.mark.parametrize("sampled,accepted", [
    ("2026-09-24T09:45:00", True), ("2026-09-24T12:00:00", True), ("2026-09-24T15:45:00", True),
    ("2026-09-24T09:44:59", False), ("2026-09-24T15:45:01", False), ("2026-09-24T09:30:00", False),
    ("2026-09-24T16:00:00", False), ("2026-09-24T08:00:00", False), ("2026-09-24T17:30:00", False),
    ("2026-09-24T00:00:00", False)])
def test_the_capture_window_is_0945_to_1545_eastern_inclusive_at_both_ends(sampled, accepted):
    assert (_hours_refusal({"sampled_at_local": sampled}) is None) is accepted


@pytest.mark.parametrize("state,accepted", [
    ("open", True), ("OPEN", True), ("Open", True), ("open (regular session)", True), (" open", True),
    ("closed", False), ("Closed", False), ("pre-market", False), ("after-hours", False),
    ("market closed", False), ("", False), ("unknown", False)])
def test_the_market_state_must_start_with_open_in_any_case(state, accepted):
    # the time and date are valid here, so ONLY the state can refuse
    assert (_hours_refusal({"market_state": state}) is None) is accepted


@pytest.mark.parametrize("sampled", ["2026-09-23T10:00:00", "2026-09-25T10:00:00", "2025-09-24T10:00:00",
                                     "2026-10-24T10:00:00"])
def test_the_capture_must_be_dated_the_run_date(sampled):
    assert _hours_refusal({"sampled_at_local": sampled}) is not None      # time and state are valid


@pytest.mark.parametrize("sampled", ["not-a-time", "", "NaT", "2026-13-45T10:00:00", 12345, ["2026-09-24T10:00:00"]])
def test_an_unparseable_sample_time_is_a_refusal_not_an_exception(sampled):
    assert _hours_refusal({"sampled_at_local": sampled}) is not None


@pytest.mark.parametrize("missing", ["sampled_at_local", "market_state"])
def test_a_payload_missing_either_field_is_a_refusal(missing):
    assert _hours_refusal({missing: None}) is not None


@pytest.mark.parametrize("sampled,accepted", [
    ("2026-09-24T09:50:00Z", False),         # 05:50 Eastern: a naive read of the digits would wrongly accept it
    ("2026-09-24T13:50:00Z", True),          # 09:50 Eastern
    ("2026-09-24T19:45:00+00:00", True),     # 15:45 Eastern, the inclusive close
    ("2026-09-24T19:46:00Z", False),         # 15:46 Eastern
    ("2026-09-24T10:00:00-04:00", True),     # already Eastern daylight time
    ("2026-09-25T01:00:00Z", False)])        # the evening of the 24th Eastern
def test_a_timezone_aware_sample_is_converted_to_eastern_and_a_naive_one_is_read_as_eastern(sampled, accepted):
    assert (_hours_refusal({"sampled_at_local": sampled}) is None) is accepted


def test_the_refusal_names_what_was_wrong():
    assert "closed" in _hours_refusal({"market_state": "closed"})
    assert "2026-09-23" in _hours_refusal({"sampled_at_local": "2026-09-23T10:00:00"})
    assert "09:45" in _hours_refusal({"sampled_at_local": "2026-09-24T09:00:00"})


@pytest.mark.parametrize("overrides", [
    {"market_state": "closed"},                                       # closed, time and date valid
    {"market_state": "pre-market", "sampled_at_local": "2026-09-24T09:00:00"},
    {"market_state": "after-hours", "sampled_at_local": "2026-09-24T17:00:00"},
    {"sampled_at_local": "2026-09-23T10:00:00"},                      # wrong date
    {"sampled_at_local": "yesterday morning"},                        # unparseable
    {"sampled_at_local": "2026-09-24T15:46:00"},                      # a minute after the window
    {"sampled_at_local": "2026-09-24T09:44:00"}])                     # a minute before it
def test_a_real_run_off_hours_is_refused_with_a_reason_code_and_writes_nothing(tmp_path, overrides):
    result, ledger_path = _call(tmp_path, dry_run=False, synthetic=False, quotes_overrides=overrides)
    assert result["reason_code"] == "quotes_not_market_hours" and result["family_id"] == _FAMILY
    assert result["reason"] and "verdict" not in result and "stage" not in result
    assert not os.path.exists(ledger_path)
    assert not os.path.exists(str(tmp_path / "research"))           # no OHLC cache either: nothing was fetched


@pytest.mark.parametrize("overrides,field", [({"sampled_at_local": None}, "sampled_at_local"),
                                             ({"market_state": None}, "market_state"),
                                             ({"sampled_at_local": 12345}, "sampled_at_local")])
def test_a_real_run_whose_capture_lacks_a_metadata_field_is_reason_coded_like_an_off_hours_one(
        tmp_path, overrides, field):
    result, ledger_path = _call(tmp_path, dry_run=False, synthetic=False, quotes_overrides=overrides)
    assert result["reason_code"] == "quotes_not_market_hours" and result["family_id"] == _FAMILY
    assert field in result["reason"] and str(tmp_path) not in result["reason"]      # names the field, never a path
    assert "verdict" not in result and "stage" not in result
    assert not os.path.exists(ledger_path)
    assert not os.path.exists(str(tmp_path / "research"))


def test_a_dry_run_whose_capture_lacks_a_metadata_field_still_raises_from_the_shared_loader(tmp_path):
    with pytest.raises(ValueError, match="sampled_at_local"):
        _call(tmp_path, dry_run=True, synthetic=True, quotes_overrides={"sampled_at_local": None})


def test_a_dry_run_is_exempt_from_the_market_hours_guard(tmp_path):
    result, ledger_path = _call(tmp_path, dry_run=True, synthetic=True,
                                quotes_overrides={"market_state": "SYNTHETIC-closed",
                                                  "sampled_at_local": "2026-01-02T03:00:00"})
    assert "reason_code" not in result and result["verdict"]
    assert not os.path.exists(ledger_path)


@pytest.mark.parametrize("sampled", ["2026-09-24T09:45:00", "2026-09-24T15:45:00"])
def test_a_real_run_at_either_edge_of_the_window_is_scored(tmp_path, sampled):
    result, ledger_path = _call(tmp_path, dry_run=False, synthetic=False, world=_low_end_world(),
                                quotes_overrides={"sampled_at_local": sampled})
    assert "reason_code" not in result and result["stage"] in _SCORED
    assert len([r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]) == 1


# ---------- the guard must NEVER raise on a bad payload and
# must not let a date-less string borrow the machine clock's date

@pytest.mark.parametrize("sampled,tested", [
    ("2026-11-01T01:30:00", "2026-11-01"),    # a wall-clock time that occurs twice (fall back): AmbiguousTimeError
    ("2026-03-08T02:30:00", "2026-03-08"),    # a wall-clock time that never occurs (spring forward)
])
def test_a_dst_ambiguous_or_nonexistent_naive_time_is_a_refusal_not_an_exception(sampled, tested):
    assert _hours_refusal({"sampled_at_local": sampled}, tested=tested) is not None


@pytest.mark.parametrize("sampled", ["10:00", "10:00:00", "  10:30", "now", "today", "noon"])
def test_a_date_less_sample_is_refused_even_when_the_machine_clock_date_is_the_run_date(sampled):
    # pandas reads a bare clock time as the CURRENT date, which would pass the date check on the run date
    today = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
    assert _hours_refusal({"sampled_at_local": sampled}, tested=today) is not None


@pytest.mark.parametrize("tested", ["not-a-date", "", None, 20_260_924, "2026-13-45"],
                         ids=["word", "empty", "none", "integer", "impossible_month_and_day"])
def test_a_bad_run_date_is_a_refusal_not_an_exception(tested):
    assert _hours_refusal(tested=tested) is not None


def test_a_dry_run_that_survived_but_failed_the_controls_carries_the_flag_and_is_not_a_pre_registered_pass(
        tmp_path, monkeypatch):
    _force_main_verdict_survived(monkeypatch)
    _force_signal_evidence(monkeypatch, False)
    result, ledger_path = _call(tmp_path, dry_run=True, synthetic=True, world=_low_end_world())
    assert result["verdict"] == "survived"
    assert result["pre_registered_pass"] is False
    assert result["pre_registered_clause"] == _FAILED_FLAG
    assert not os.path.exists(ledger_path)                                   # a dry run still writes nothing


def test_the_dry_run_cli_prints_the_flag_when_survived_but_the_controls_failed(tmp_path, monkeypatch, capsys):
    _force_main_verdict_survived(monkeypatch)
    _force_signal_evidence(monkeypatch, False)
    world = _low_end_world()
    catalog_path, quotes_path, ledger_path, research_root, futures_dir = _fixture(tmp_path, world=world)
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(world))
    assert driver.main(["--dry-run", "--quotes", quotes_path, "--catalog", catalog_path, "--ledger", ledger_path,
                        "--research-root", research_root, "--futures-dir", futures_dir, "--random-draws", "3"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert _FAILED_FLAG in lines[:-1]
    printed = json.loads(lines[-1])["result"]
    assert printed["pre_registered_pass"] is False and printed["pre_registered_clause"] == _FAILED_FLAG
    assert not os.path.exists(ledger_path)


# ---------- tax wording, long-term window counts, the untaxed-basis
# comparison, the holdout HIGH share, the module hash, a reason-coded zero-variance refusal

def _long_episode_level_path(n_months):
    level = np.full(n_months, 5.0)
    level[65:82] = 5.6          # a 17-month HIGH episode: one merged window held over 365 days
    level[82:90] = 4.9
    level[90:96] = 5.6
    level[96:] = 4.8
    return level


def _late_high_level_path(n_months):
    level = np.full(n_months, 5.0)
    level[65:76] = 5.6
    level[76:86] = 4.9
    level[86:96] = 5.6
    level[96:125] = 4.8
    level[125:140] = 5.6        # a HIGH block inside the last three years (the ABI-1 holdout window)
    level[140:] = 4.8
    return level


def _late_high_world(monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "_level_path", _late_high_level_path)
    return _low_end_world()


def _long_window_world(monkeypatch):
    """The low-end world (no window open at the data's end) whose first HIGH episode is long enough for a
    merged window to be held more than 365 calendar days."""
    monkeypatch.setattr(sys.modules[__name__], "_level_path", _long_episode_level_path)
    return _low_end_world()


def _window_days(windows, last_bar):
    """Calendar days held by each window as the engine trades it: a window entered on the last bar opens
    nothing, one still open at the data's end closes at the last bar."""
    return [(min(exit_, last_bar) - entry).days for entry, exit_ in windows if entry < last_bar]


def _lt_stats(days):
    long_term = [d for d in days if d > 365]
    return {"n_windows": len(days), "n_long_term": len(long_term), "window_days": sum(days),
            "long_term_window_days": sum(long_term)}


def _real_note(tmp_path, world, **kwargs):
    result, ledger_path = _call(tmp_path, dry_run=False, synthetic=False, world=world, **kwargs)
    outcomes = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    assert len(outcomes) == 1
    return result, outcomes[0]["note"]


def test_the_long_term_window_stats_count_only_windows_held_strictly_more_than_365_days():
    trades = pd.DataFrame({
        "entry_date": pd.to_datetime(["2010-01-04", "2011-01-03", "2012-01-03", "2013-01-02"]),
        "exit_date": pd.to_datetime(["2011-01-04", "2012-01-04", "2012-02-02", "2013-01-03"])})
    stats = driver.long_term_window_stats(trades)          # held 365, 366, 30 and 1 calendar days
    assert stats["threshold_days"] == constants.LONG_TERM_HOLDING_DAYS == 365
    assert (stats["n_windows"], stats["n_long_term"]) == (4, 1)                 # exactly 365 is still short-term
    assert (stats["window_days"], stats["long_term_window_days"]) == (762, 366)
    assert stats["long_term_day_share"] == pytest.approx(366 / 762)


def test_the_long_term_window_stats_of_no_windows_are_zero_with_no_share():
    empty = pd.DataFrame({"entry_date": pd.to_datetime([]), "exit_date": pd.to_datetime([])})
    stats = driver.long_term_window_stats(empty)
    assert (stats["n_windows"], stats["n_long_term"], stats["window_days"], stats["long_term_window_days"]) == (0, 0, 0, 0)
    assert stats["long_term_day_share"] is None


def test_the_main_always_on_and_random_arms_report_their_long_term_windows_from_their_own_trades(tmp_path, monkeypatch):
    world = _long_window_world(monkeypatch)
    result, _ = _call(tmp_path, world=world, n_random_draws=4)
    ctx = _prepared(world)
    last_bar = world["SPY"].index[-1]
    expected = _lt_stats(_window_days(mod.high_windows(ctx["month_ends"], ctx["states"]), last_bar))
    assert expected["n_long_term"] >= 1 and expected["n_long_term"] < expected["n_windows"]     # a discriminating fixture
    got = result["overlay_diagnostics"]["long_term_windows"]
    assert {k: got[k] for k in expected} == expected and got["threshold_days"] == 365
    assert got["long_term_day_share"] == pytest.approx(expected["long_term_window_days"] / expected["window_days"])
    always = _lt_stats(_window_days(mod.high_windows(ctx["month_ends"], [True] * len(ctx["states"])), last_bar))
    a = result["controls"]["always_on"]["long_term_windows"]
    assert {k: a[k] for k in always} == always and always["n_windows"] == always["n_long_term"] == 1
    draws = mod.random_window_state_draws(ctx["states"], 4, seed=mod.RANDOM_CONTROL_SEED)
    per_draw = [_lt_stats(_window_days(mod.high_windows(ctx["month_ends"], d), last_bar)) for d in draws]
    r = result["controls"]["random_window"]["long_term_windows"]
    for key in ("n_windows", "n_long_term", "window_days", "long_term_window_days"):
        assert r[key] == [p[key] for p in per_draw], key
    assert len(r["n_long_term"]) == 4


def test_the_controls_block_prints_the_untaxed_basis_comparison_and_leaves_the_rule_unchanged(tmp_path):
    result, _ = _call(tmp_path, n_random_draws=4)
    controls = result["controls"]
    untaxed = controls["untaxed_basis_comparison"]
    assert untaxed["main_sharpe_untaxed"] == result["overlay_diagnostics"]["sharpe_daily_excess_untaxed"]
    assert untaxed["always_on_sharpe_untaxed"] == controls["always_on"]["sharpe_daily_excess_untaxed"]
    assert untaxed["main_exceeds_always_on_untaxed"] is bool(
        untaxed["main_sharpe_untaxed"] > untaxed["always_on_sharpe_untaxed"])
    assert "tax-asymmetric" in untaxed["note"]
    assert set(controls["interpretation"]) == {"signal_evidence", "percentile_bar", "rule"}      # the rule is unchanged
    assert controls["interpretation"] == driver.interpret_controls(
        main_sharpe=result["sharpe_daily_excess"], always_on_sharpe=controls["always_on"]["sharpe_daily_excess"],
        main_percentile=controls["random_window"]["main_percentile"])
    assert "no modelled tax" in controls["always_on"]["note"]


def test_the_ledger_note_states_the_always_on_arm_pays_no_modelled_tax_and_lists_the_untaxed_windows(tmp_path):
    result, note = _real_note(tmp_path, _low_end_world())
    assert "pays no modelled tax (long-term at the 0.0 rate), not merely deferred" in note
    flatters = note.index("each of these flatters the strategy")
    flat_rate = note.index("windows held over 365 days are untaxed at the pipeline's 0.0 long-term rate")
    assert flatters < flat_rate < note.index("the tax walk (ABI-1 realization netting)")           # inside the list
    assert "tax-asymmetric" in note
    assert "pre_registered_pass=false" in note.lower() or "pre_registered_pass=true" in note.lower()


def test_the_ledger_note_prints_the_long_term_window_counts_computed_from_this_runs_trades(tmp_path, monkeypatch):
    world = _long_window_world(monkeypatch)
    result, note = _real_note(tmp_path, world)
    ctx = _prepared(world)
    last_bar = world["SPY"].index[-1]
    main = _lt_stats(_window_days(mod.high_windows(ctx["month_ends"], ctx["states"]), last_bar))
    always = _lt_stats(_window_days(mod.high_windows(ctx["month_ends"], [True] * len(ctx["states"])), last_bar))
    assert main["n_long_term"] >= 1
    assert (f"main arm {main['n_long_term']} of {main['n_windows']} windows and {main['long_term_window_days']} of "
            f"{main['window_days']} window calendar days") in note
    assert (f"always-on arm {always['n_long_term']} of {always['n_windows']} windows and "
            f"{always['long_term_window_days']} of {always['window_days']} window calendar days") in note
    draws = mod.random_window_state_draws(ctx["states"], 6, seed=mod.RANDOM_CONTROL_SEED)     # the suite's 6-draw run
    per_draw = [_lt_stats(_window_days(mod.high_windows(ctx["month_ends"], d), last_bar)) for d in draws]
    counts = [p["n_long_term"] for p in per_draw]
    assert (f"random-window arms (mean [min, max] over 6 draws) {np.mean(counts):.2f} [{min(counts)}, {max(counts)}] "
            f"windows over 365 days of {np.mean([p['n_windows'] for p in per_draw]):g}") in note
    assert "held more than 365 calendar days" in note and "counted from this run's own trades" in note


def test_the_ledger_note_prints_the_untaxed_basis_comparison_beside_the_taxed_rule(tmp_path):
    result, note = _real_note(tmp_path, _low_end_world())
    match = re.search(r"main_sharpe_untaxed=(\S+); always_on_sharpe_untaxed=(\S+); "
                      r"main_exceeds_always_on_untaxed=(True|False|None)", note)
    assert match, "the untaxed-basis comparison is missing from the note"
    diag = result["overlay_diagnostics"]
    always = result["controls"]["always_on"]["sharpe_daily_excess_untaxed"]
    assert float(match.group(1)) == pytest.approx(round(diag["sharpe_daily_excess_untaxed"], 4))
    assert float(match.group(2)) == pytest.approx(round(always, 4))
    assert match.group(3) == str(bool(diag["sharpe_daily_excess_untaxed"] > always))


def test_the_holdout_window_high_share_is_computed_from_this_runs_own_states(tmp_path):
    world = _low_end_world()
    result, note = _real_note(tmp_path, world)
    ctx = _prepared(world)
    sessions = world["SPY"].index[world["SPY"].index >= ctx["first_decision"]]
    n_holdout = result["window"]["holdout_sessions"]
    start = sessions[-n_holdout]
    in_holdout = [i for i, d in enumerate(ctx["decisions"]) if d >= start]
    n, high = len(in_holdout), sum(bool(ctx["states"][i]) for i in in_holdout)
    assert n >= 20                                                    # a real span of month-ends
    got = result["overlay_diagnostics"]["holdout_window_high"]
    assert (got["start"], got["n_month_ends"], got["n_high"]) == (start.strftime("%Y-%m-%d"), n, high)
    assert got["high_share"] == pytest.approx(high / n)
    assert f"holdout window (from {start:%Y-%m-%d}) holds {high} HIGH of {n} month-ends (share {round(high / n, 4)})" in note


def test_the_holdout_window_share_follows_the_states_it_is_computed_from(tmp_path, monkeypatch):
    """A second world (a different HIGH pattern in the holdout) must give a different printed share."""
    a_world = _low_end_world()
    _, note_a = _real_note(_mk(tmp_path, "a"), a_world)
    b_world = _late_high_world(monkeypatch)
    _, note_b = _real_note(_mk(tmp_path, "b"), b_world)
    share = lambda text: re.search(r"holds (\d+) HIGH of (\d+) month-ends \(share ([0-9.]+)\)", text).groups()  # noqa: E731
    assert share(note_a) != share(note_b)


def test_the_ledger_note_records_the_signal_modules_own_sha256_beside_the_drivers(tmp_path):
    result, note = _real_note(tmp_path, _low_end_world())
    assert f"signal_module_sha256={evidence.source_sha256(mod.__file__)}" in note
    assert f"driver_sha256={'a' * 64}" in note
    assert evidence.source_sha256(mod.__file__) != "a" * 64


def test_the_ledger_note_describes_the_close_as_consistent_with_the_settlement_and_states_the_calendar_direction(tmp_path):
    result, note = _real_note(tmp_path, _low_end_world())
    assert "consistent with the COMEX settlement time" in note
    assert "favourable to the DSR" in note                                # the account-calendar slice, with its direction


def test_the_driver_docstring_carries_the_flattering_tax_assumption_and_the_market_hours_guard():
    text = " ".join(driver.__doc__.split())
    assert "windows held over 365 days are untaxed at the pipeline's 0.0 long-term rate" in text
    assert "quotes_not_market_hours" in text and "09:45 to 15:45 Eastern" in text
    assert "pays no modelled tax (long-term at the 0.0 rate), not merely deferred" in text


def test_a_zero_variance_futures_history_is_a_reason_coded_refusal_at_the_driver(tmp_path):
    world = _world()
    world["gc"] = world["hg"] * np.exp(5.0)                            # the log ratio never varies
    result, ledger_path = _call(tmp_path, world=world)
    assert result["reason_code"] == "futures_window_zero_variance" and "verdict" not in result
    assert not os.path.exists(ledger_path)


# ----------------------------------------------------------------- guards

def test_real_run_refuses_a_synthetic_quotes_file(tmp_path):
    with pytest.raises(ValueError, match="SYNTHETIC"):
        _call(tmp_path, dry_run=False, synthetic=True)


def test_a_missing_spy_quote_is_refused(tmp_path):
    world = _world()
    catalog_path, quotes_path, ledger_path, research_root, futures_dir = _fixture(tmp_path, world=world)
    payload = _quotes_payload(synthetic=True)
    del payload["quotes"]["SPY"]
    with open(quotes_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    with pytest.raises(ValueError, match="SPY"):
        driver.compute_and_record(
            catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path, run_id="x",
            tested_date="2026-09-24", dry_run=True, signal_source_path="p", signal_source_sha256="s",
            research_root=research_root, ohlc_loader=_loader(world), futures_dir=futures_dir, n_random_draws=2)


def test_missing_catalog_entry_and_ineligible_status_skip_cleanly(tmp_path):
    result, _ = _call(_mk(tmp_path, "inelig"), status="ruled_out")
    assert result["reason_code"] == "catalog_status"
    world = _world()
    catalog_path, quotes_path, ledger_path, research_root, futures_dir = _fixture(_mk(tmp_path, "none"), world=world)
    with open(catalog_path, "w", encoding="utf-8") as fh:
        json.dump([], fh)
    skipped = driver.compute_and_record(
        catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path, run_id="x",
        tested_date="2026-09-24", dry_run=True, signal_source_path="p", signal_source_sha256="s",
        research_root=research_root, ohlc_loader=_loader(world), futures_dir=futures_dir, n_random_draws=2)
    assert skipped["reason_code"] == "not_in_catalog"


def test_parameter_drift_blocks_the_run(tmp_path):
    with open(_ENTRY_PATH, "r", encoding="utf-8") as fh:
        drifted = dict(_catalog_entry_from(fh)["parameters"])
    drifted["z_window_month_ends"] = 48
    result, _ = _call(tmp_path, parameters=drifted)
    assert result["reason_code"] == "parameter_drift"


def test_an_existing_abi1_outcome_blocks_a_second_trial(tmp_path):
    world = _world()
    catalog_path, quotes_path, ledger_path, research_root, futures_dir = _fixture(tmp_path, world=world)
    trial_ledger.record_outcome(ledger_path, _FAMILY, "earlier", "2026-09-01", stage="cut_dsr", dsr_value=0.1,
                                sharpe_in_sample=0.0, construction="ABI-1", note="prior")
    result = driver.compute_and_record(
        catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path, run_id="again",
        tested_date="2026-09-24", dry_run=True, signal_source_path="p", signal_source_sha256="s",
        research_root=research_root, ohlc_loader=_loader(world), futures_dir=futures_dir, n_random_draws=2)
    assert result["reason_code"] == "already_scored_abi1"


# ------------------------------------------------------ ledger path (tmp ledger)

def test_a_non_dry_run_against_a_temp_ledger_writes_exactly_one_abi1_row_with_disclosures(tmp_path):
    world = _low_end_world()
    result, ledger_path = _call(tmp_path, dry_run=False, synthetic=False, world=world)
    rows = trial_ledger.read_all(ledger_path)
    outcomes = [r for r in rows if r.get("event") == "outcome"]
    assert len(outcomes) == 1 and sorted(r.get("event") for r in rows) == ["first_test", "outcome"]
    row = outcomes[0]
    assert row["family_id"] == _FAMILY and row["construction"] == "ABI-1"
    assert result["stage"] in _SCORED                    # the fixture scores, so the note IS written
    assert row["stage"] == result["stage"]
    assert row["dsr_value"] == pytest.approx(result["dsr_value"], rel=1e-12)
    assert row["sharpe_in_sample"] == pytest.approx(result["sharpe_daily_excess"], rel=1e-12)
    note = (row.get("note") or "").lower()
    assert note.startswith("cash_convention=sweep_cash_daily; benchmark=spy; tax_policy=realization; cost model:")
    for phrase in (
            # what the account is and how it is financed and charged
            "benchmark-plus-overlay account", "extra spy", "0.25 of equity", "^irx", "no spread is charged",
            "flatters the strategy", "not modelled", "margin rate", "bill rate plus a spread",
            "full round trip", "tax walk", "different strategy definition",
            # the signal, the lag and the calendar
            "ln(gc=f close / hg=f close)", "last actual nyse session strictly before",
            "one-session lag", "60 monthly reads", "z >= 0", "scheduled", "never read off the data",
            # construction rules: merged windows, fixed shares, holes
            "merged into one window", "fixed share count", "longest_episode_months=",
            "as-of", "warm-up", "refused",
            # the settlement question and the frozen cache
            "settlement", "frozen", "sha256-verified", "futures rows=",
            # the pre-declared controls, with the measured numbers
            "control arms (report-only", "always-on", "random-window", "interpretation rule (pre-declared)",
            "signal_evidence=", "both control conditions hold", "only if the verdict is also a pass",
            "main_percentile=", "always_on_sharpe=", "untaxed",
            # the risk layer
            "risk layer (the risk-basis rule", "overlay-only excess account", "risk_account_basis=overlay_only_excess",
            "whole-account reference", "gating nothing", "spy-relative refusals", "refusals failed in this run",
            # coverage flags in the permanent note
            "final_window_truncated=false", "window_entered_on_last_bar=false",
            "accepted_open_final_window=false", "decisions=",
            # honesty
            "pre-registration limits", "not all chosen blind", "outside the trial ledger", "single pre-declared shot",
            "descriptive only", "a few independent episodes"):
        assert phrase in note, phrase
    # the calendar's provenance and the last-bar status are in the permanent note
    prov = mod.schedule_provenance(_prepared(world))
    assert f"exchange_calendars_version={prov['exchange_calendars_version']}" in note
    assert f"schedule_sha256={prov['schedule_sha256']}" in note
    assert mod.SCHEDULE_HASH_FORMAT.lower() in note
    assert f"schedule_decisions={prov['n_decisions']}" in note and f"schedule_terminal_exit={prov['terminal_exit']}" in note
    last_bar = world["SPY"].index[-1].strftime("%Y-%m-%d")
    assert f"last spy bar used is dated {last_bar}" in note and "run tested date is 2026-09-24" in note
    assert "last_spy_bar_may_be_live_partial=false" in note
    risk = result["overlay_diagnostics"]["risk_layer"]
    assert f"max drawdown {round(risk['account_max_drawdown'], 4)} against spy's {round(risk['benchmark_max_drawdown'], 4)}" in note
    controls = result["controls"]
    assert f"main_percentile={round(controls['random_window']['main_percentile'], 4)}" in note
    assert f"n_draws={controls['random_window']['n_draws']}" in note
    assert not _DOLLAR_FIGURE.search(note) and not _EIGHT_DIGITS.search(note)     # ledger notes must stay clear of these shapes too
    assert os.path.abspath(ledger_path).startswith(os.path.abspath(str(tmp_path)))


def test_an_unscoreable_real_run_records_the_instrument_set_benchmark_and_cash_convention(tmp_path):
    world = _cut(_world(), "2005-09-30")        # 45 account sessions, LOW at the last month-end: not scoreable
    result, ledger_path = _call(tmp_path, dry_run=False, synthetic=False, world=world)
    assert result["verdict"] == "not_scoreable" and result["stage"] == "insufficient_evidence"
    outcomes = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    assert len(outcomes) == 1
    observed = outcomes[0]["evidence"]["observed"]
    assert observed["instrument_set"] == ["SPY"]
    assert observed["benchmark"] == "SPY" and observed["cash_convention"] == "sweep_cash_daily"
    with open(_ENTRY_PATH, "r", encoding="utf-8") as fh:
        assert observed["parameters"] == _catalog_entry_from(fh)["parameters"]


def test_pre_declared_default_paths_are_the_real_ones_and_are_never_used_by_these_tests():
    assert os.path.abspath(driver.LEDGER_PATH).endswith(os.path.join("data", "mining", "trial_ledger.jsonl"))
    assert os.path.abspath(driver.FUTURES_DIR).endswith(os.path.join("data", "mining", "gold_copper"))
    for path in (driver.LEDGER_PATH, driver.FUTURES_DIR):
        assert os.path.abspath(path).startswith(REPO_ROOT)


# ------------------------------------------------------ the final window

def _open_window_world(past):
    """The world cut ``past`` sessions after the LAST window entry (0 = on the
    entry date, 3 = inside the open window)."""
    world = _world()
    ctx = _prepared(world)
    states = [bool(x) for x in ctx["states"]]
    sessions = world["SPY"].index
    starts = [k for k in range(1, len(states)) if states[k] and not states[k - 1]
              and sessions.get_loc(ctx["decisions"][k]) + past < len(sessions) - 1]
    entry = ctx["decisions"][max(starts)]
    last = sessions[sessions.get_loc(entry) + past]
    return _cut(world, last), entry


def test_a_dry_run_reports_a_truncated_final_window_and_the_note_carries_it(tmp_path):
    world, entry = _open_window_world(3)
    result, _ = _call(tmp_path, world=world)
    diag = result["overlay_diagnostics"]
    assert diag["final_window_truncated"] is True and diag["window_entered_on_last_bar"] is False
    assert result["account_end"] == world["SPY"].index[-1].strftime("%Y-%m-%d")
    assert diag["last_window_exit"] == world["SPY"].index[-1].strftime("%Y-%m-%d")


def test_a_real_run_refuses_a_truncated_final_window_unless_it_is_explicitly_accepted(tmp_path):
    """A state-driven window is ordinarily open at the data's end, so the refusal
    is the DEFAULT, not a hard stop: the closing bar could be a live partial bar."""
    world, _ = _open_window_world(3)
    result, ledger_path = _call(tmp_path, dry_run=False, synthetic=False, world=world)
    assert result["reason_code"] == "truncated_final_window"
    assert "--accept-open-final-window" in result["reason"]
    assert "verdict" not in result and not os.path.exists(ledger_path)


def test_an_accepted_open_final_window_is_scored_and_recorded_in_the_note(tmp_path):
    world, _ = _open_window_world(3)
    result, ledger_path = _call(tmp_path, dry_run=False, synthetic=False, world=world, accept_open_final_window=True)
    assert result["stage"] in _SCORED + ("insufficient_evidence",)
    outcomes = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    note = outcomes[0]["note"].lower()
    assert "final_window_truncated=true" in note and "accepted_open_final_window=true" in note
    assert "closed at the last close" in note


@pytest.mark.parametrize("bar,tested,partial", [
    ("2026-09-23", "2026-09-24", False),        # the last bar is a prior, completed session
    ("2026-09-24", "2026-09-24", True),         # a bar dated the run date can be the vendor's in-progress session
    ("2026-09-25", "2026-09-24", True),         # a bar AFTER the stated run date: inconsistent, treated as possibly partial
])
def test_the_last_bar_clause_compares_the_last_bar_date_with_the_run_date(bar, tested, partial):
    """Whether the last SPY bar may be a live partial bar is decided
    by comparing its date with the run's tested date, and the note says which."""
    may_be_partial, text = driver.last_bar_partial_clause(bar, tested)
    assert may_be_partial is partial
    assert f"last_spy_bar_may_be_live_partial={partial}" in text
    assert f"last SPY bar used is dated {bar}" in text and f"run tested date is {tested}" in text
    assert ("may be the vendor's live partial bar" in text) is partial


@pytest.mark.parametrize("bad", ["not-a-date", "", None])
def test_a_malformed_tested_date_is_refused_before_any_work_is_done(tmp_path, bad):
    """The tested date drives a sentence in the permanent note, so a malformed one
    must fail at the door, not minutes into the control arms (the catalog path below
    does not exist: any work would fail differently)."""
    with pytest.raises(ValueError, match="tested_date"):
        driver.compute_and_record(
            catalog_path=str(tmp_path / "missing.json"), quotes_path=str(tmp_path / "missing-quotes.json"),
            ledger_path=str(tmp_path / "ledger.jsonl"), run_id="x", tested_date=bad, dry_run=True,
            signal_source_path="p", signal_source_sha256="s", research_root=str(tmp_path / "research"))
    assert not os.path.exists(str(tmp_path / "ledger.jsonl"))


def test_a_run_dated_the_last_bar_is_flagged_as_possibly_resting_on_a_partial_bar(tmp_path):
    world = _low_end_world()
    last_bar = world["SPY"].index[-1].strftime("%Y-%m-%d")
    same_day, _ = _call(_mk(tmp_path, "same_day"), world=world, tested_date=last_bar)
    later, _ = _call(_mk(tmp_path, "later"), world=world, tested_date=(world["SPY"].index[-1] + pd.Timedelta(days=1)
                                                                       ).strftime("%Y-%m-%d"))
    assert "last_spy_bar_may_be_live_partial=True" in same_day["cost_model_note"]
    assert "last_spy_bar_may_be_live_partial=False" in later["cost_model_note"]
    assert "last_spy_bar_may_be_live_partial=True" not in later["cost_model_note"]


def test_data_ending_on_an_entry_date_opens_nothing_and_needs_no_acceptance(tmp_path):
    world, entry = _open_window_world(0)
    result, ledger_path = _call(tmp_path, dry_run=False, synthetic=False, world=world)
    assert "reason_code" not in result
    diag = result["overlay_diagnostics"]
    assert diag["window_entered_on_last_bar"] is True and diag["final_window_truncated"] is False
    outcomes = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    assert "window_entered_on_last_bar=true" in outcomes[-1]["note"].lower()


# ------------------------------------------------ the frozen futures cache

def test_a_tampered_futures_cache_is_refused_with_a_reason_code_and_writes_nothing(tmp_path):
    catalog_path, quotes_path, ledger_path, research_root, futures_dir = _fixture(tmp_path)
    path = os.path.join(futures_dir, "yahoo_GC_F.csv")
    with open(path, "rb") as fh:
        raw = fh.read()
    with open(path, "wb") as fh:
        fh.write(raw.replace(b"2005-07-28", b"2005-07-27", 1))
    result, _ = _call(_mk(tmp_path, "run"), futures_dir=futures_dir, dry_run=False, synthetic=False)
    assert result["reason_code"] == "futures_cache_hash_mismatch"
    assert "verdict" not in result


def test_a_missing_futures_cache_is_refused_with_a_reason_code(tmp_path):
    result, _ = _call(tmp_path, futures=False)
    assert result["reason_code"] == "futures_cache_missing"


def test_a_futures_hole_after_the_first_defined_z_is_refused_naming_the_date(tmp_path):
    world = _world()
    ctx = _prepared(world)
    read = ctx["table"].loc[pd.Timestamp("2008-03-31"), "read_date"]
    holed = dict(world)
    holed["gc"] = world["gc"].drop(index=read)
    result, _ = _call(tmp_path, world=holed)
    assert result["reason_code"] == "futures_read_hole"
    assert read.strftime("%Y-%m-%d") in result["missing_dates"]
    assert "verdict" not in result


def test_a_futures_history_too_short_to_define_z_is_refused_with_a_reason_code(tmp_path):
    result, _ = _call(tmp_path, world=_cut(_world(), "2004-06-30"))
    assert result["reason_code"] == "futures_history_too_short" and "verdict" not in result


def test_a_frozen_futures_cache_that_ends_before_the_last_read_is_refused_as_stale(tmp_path):
    world = _world()
    catalog_path, quotes_path, ledger_path, research_root, futures_dir = _fixture(tmp_path, world=world, futures=False)
    _freeze(futures_dir, world, through="2013-10-31")           # SPY runs to 2013-12-31: the Nov and Dec reads are absent
    result, _ = _call(_mk(tmp_path, "run"), world=world, futures_dir=futures_dir)
    assert result["reason_code"] == "futures_read_hole"
    assert any(d >= "2013-11-01" for d in result["missing_dates"])


# ---------------------------------- data-calendar gaps are refused, not trimmed

@pytest.mark.parametrize("dry_run", [True, False])
def test_a_missing_exchange_session_in_spy_is_refused_and_named(tmp_path, dry_run):
    world = _world()
    hole = world["SPY"].index[1500]
    world["SPY"] = world["SPY"].drop(index=hole)
    result, ledger_path = _call(tmp_path, world=world, dry_run=dry_run, synthetic=dry_run)
    assert result["reason_code"] == "missing_exchange_session"
    assert hole.strftime("%Y-%m-%d") in result["missing_dates"] and result["n_missing_dates"] == 1
    assert "verdict" not in result and not os.path.exists(ledger_path)


def test_a_stray_bar_on_a_closed_day_is_refused(tmp_path):
    world = _world()
    extra = world["SPY"].iloc[[1000]].copy()
    extra.index = pd.DatetimeIndex([pd.Timestamp("2008-12-25")])
    world["SPY"] = pd.concat([world["SPY"], extra]).sort_index()
    result, _ = _call(tmp_path, world=world)
    assert result["reason_code"] == "bar_on_closed_day" and "2008-12-25" in result["extra_dates"]


def test_a_spy_history_that_starts_after_the_first_decision_is_refused(tmp_path):
    world = _world(spy_start="2006-01-03")
    result, _ = _call(tmp_path, world=world)
    assert result["reason_code"] == "spy_history_starts_after_first_decision"


# ------------------------------------------------- ohlc cache provenance

def test_a_real_run_refuses_ohlc_cached_by_an_earlier_dry_run_under_the_same_run_id(tmp_path):
    world = _world()
    _call(_mk(tmp_path, "dry"), world=world, run_id="shared")
    catalog_path, quotes_path, ledger_path, research_root, futures_dir = _fixture(
        _mk(tmp_path, "real"), synthetic=False, world=world)
    shutil_root = os.path.join(str(tmp_path / "dry"), "research", "shared")
    result = driver.compute_and_record(
        catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path, run_id="shared",
        tested_date="2026-09-24", dry_run=False, signal_source_path="p", signal_source_sha256="s",
        research_root=os.path.join(str(tmp_path / "dry"), "research"), ohlc_loader=_loader(world),
        futures_dir=futures_dir, n_random_draws=driver.DEFAULT_RANDOM_DRAWS)
    assert os.path.isdir(shutil_root)
    assert result["reason_code"] == "stale_ohlc_cache"
    assert sorted(result["cached_symbols"]) == ["SPY", "^IRX"]
    assert not os.path.exists(ledger_path)


def test_a_real_run_with_a_fresh_run_id_after_a_dry_run_is_scored_from_fetched_history(tmp_path):
    world = _world()
    _call(_mk(tmp_path, "dry"), world=world, run_id="first")
    result, _ = _call(_mk(tmp_path, "real"), world=world, dry_run=False, synthetic=False, run_id="second")
    assert "reason_code" not in result and result["ohlc_sources"] == {"SPY": "fetched", "^IRX": "fetched"}


def test_result_reports_the_provenance_of_every_price_history_and_the_frozen_futures(tmp_path):
    world = _world()
    result, _ = _call(tmp_path, world=world)
    assert result["ohlc_sources"] == {"SPY": "fetched", "^IRX": "fetched"}
    assert result["last_bar_dates"]["SPY"] == world["SPY"].index[-1].strftime("%Y-%m-%d")
    assert result["last_bar_dates"]["^IRX"] == world["^IRX"].index[-1].strftime("%Y-%m-%d")
    futures = result["futures_provenance"]
    assert futures["through"] == world["gc"].index[-1].strftime("%Y-%m-%d")
    assert futures["files"]["yahoo_GC_F.csv"]["rows"] == len(world["gc"])
    assert futures["files"]["yahoo_HG_F.csv"]["sha256"] and futures["sha256_verified"] is True
    assert isinstance(result["overlay_diagnostics"]["asof_reads"], list)


# ---------------------------------------------------------------- the CLI

def test_cli_dry_run_writes_a_json_result_and_exits_zero(tmp_path, capsys, monkeypatch):
    world = _world()
    catalog_path, quotes_path, ledger_path, research_root, futures_dir = _fixture(tmp_path, world=world)
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(world))
    code = driver.main(["--dry-run", "--quotes", quotes_path, "--catalog", catalog_path, "--ledger", ledger_path,
                        "--research-root", research_root, "--futures-dir", futures_dir, "--random-draws", "3"])
    assert code == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["output"].endswith("gold-copper-overlay-dry-run.json")
    with open(os.path.join(research_root, driver.DEFAULT_RUN_ID, "gold-copper-overlay-dry-run.json"), encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["dry_run"] is True and payload["construction"] == "ABI-1"
    assert payload["result"]["family_id"] == _FAMILY
    assert not os.path.exists(ledger_path)


def _cli_paths(tmp_path):
    """Every path option pointed at tmp_path, so a CLI test can never reach a real default."""
    return ["--ledger", str(tmp_path / "cli_ledger.jsonl"), "--catalog", str(tmp_path / "cli_catalog.json"),
            "--research-root", str(tmp_path / "cli_research"), "--futures-dir", str(tmp_path / "cli_futures")]


def test_a_real_run_requires_an_explicit_tested_date_and_run_id(tmp_path):
    with pytest.raises(SystemExit):
        driver.main(["--quotes", str(tmp_path / "q.json"), "--run-id", "x"] + _cli_paths(tmp_path))
    with pytest.raises(SystemExit):
        driver.main(["--quotes", str(tmp_path / "q.json"), "--tested-date", "2026-09-24"] + _cli_paths(tmp_path))


def test_a_real_run_cannot_change_the_random_draw_count_on_the_command_line(tmp_path):
    with pytest.raises(SystemExit):
        driver.main(["--quotes", str(tmp_path / "q.json"), "--run-id", "x", "--tested-date", "2026-09-24",
                     "--random-draws", "10"] + _cli_paths(tmp_path))
    assert not (tmp_path / "cli_research").exists() and not (tmp_path / "cli_ledger.jsonl").exists()


def test_the_freeze_command_writes_only_into_the_given_directory(tmp_path, capsys, monkeypatch):
    idx = _sessions("2000-08-30", "2001-12-31")
    world = {"gc": _walk(idx, 3, 0.008, 0.0, 300.0), "hg": _walk(idx, 4, 0.012, 0.0, 90.0)}

    def fetch(symbol):
        frame = _ohlcv(world["gc"] if symbol == "GC=F" else world["hg"])
        frame.index = pd.DatetimeIndex(frame.index).tz_localize("America/New_York")
        return frame
    monkeypatch.setattr(driver.mod, "real_yahoo_fetcher", fetch)
    target = str(tmp_path / "frozen")
    code = driver.main(["--freeze-futures", "--through", "2001-12-31", "--futures-dir", target])
    assert code == 0
    assert sorted(os.listdir(target)) == ["manifest.json", "yahoo_GC_F.csv", "yahoo_HG_F.csv"]
    with pytest.raises(FileExistsError):
        driver.main(["--freeze-futures", "--through", "2001-12-31", "--futures-dir", target])
    assert driver.main(["--freeze-futures", "--through", "2001-12-31", "--futures-dir", target,
                        "--overwrite-frozen"]) == 0


def test_the_freeze_command_requires_an_explicit_through_date(tmp_path, monkeypatch):
    def refuse_network(symbol):
        raise AssertionError("a freeze without --through must not fetch anything")
    monkeypatch.setattr(driver.mod, "real_yahoo_fetcher", refuse_network)
    with pytest.raises(SystemExit):
        driver.main(["--freeze-futures", "--futures-dir", str(tmp_path / "f")])
    assert not (tmp_path / "f").exists()


# ----------------------------------------------- write call sites (redirect guard)

_WRITE_CALLS = {"os.replace", "os.makedirs", "handle.write", "clean.to_csv", "json.dump",
                "batch3._atomic_write_text", "abi1_driver.record_abi1_outcome", "mod.freeze_futures",
                "trial_ledger.record_outcome", "trial_ledger.record_first_test",
                "batch5.load_frame"}       # load_frame writes the OHLC cache under research-root/run-id


def _write_call_sites(path):
    """(enclosing function, call) for every call that can write a file: open() in
    a write mode, and the named writers. Found by AST, so a renamed variable or a
    new call site cannot slip past a text pattern."""
    import ast
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    sites = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            name = ast.unparse(node.func)
            if name == "open":
                mode = node.args[1] if len(node.args) > 1 else None
                if isinstance(mode, ast.Constant) and any(c in str(mode.value) for c in "wax"):
                    sites.add((fn.name, "open(write)"))
            elif name in _WRITE_CALLS or name.endswith((".write_text", ".write_bytes")):
                sites.add((fn.name, name))
    return sites


def test_every_file_write_in_the_driver_is_a_known_call_site_that_these_tests_redirect():
    """Every write call site is listed here, and every path it can write to is a
    parameter (--ledger, --research-root, --futures-dir) that the tests point at
    tmp_path. A new write call site fails this test until it has been reviewed."""
    assert _write_call_sites(driver.__file__) == {
        ("compute_and_record", "abi1_driver.record_abi1_outcome"),        # the --ledger path
        ("main", "os.makedirs"), ("main", "batch3._atomic_write_text"),   # under --research-root/run-id
        ("main", "mod.freeze_futures"),                                   # under --futures-dir only
        ("compute_and_record", "batch5.load_frame"),                      # writes the OHLC cache under research-root/run-id
    }


def test_every_file_write_in_the_module_is_the_frozen_cache_writer_only():
    assert _write_call_sites(mod.__file__) == {
        ("_atomic_write_bytes", "open(write)"), ("_atomic_write_bytes", "handle.write"),
        ("_atomic_write_bytes", "os.replace"), ("freeze_futures", "os.makedirs"),
        ("_serialize", "clean.to_csv"),           # returns a string: no path, nothing is written here
    }


def test_no_dollar_figure_or_yyyymmdd_literal_in_the_driver_source():
    with open(driver.__file__, "r", encoding="utf-8") as fh:
        source = fh.read()
    assert not _DOLLAR_FIGURE.search(source) and not _EIGHT_DIGITS.search(source)


# ---------- the metadata-refusal mapping must key on the loader's exact message
# shape, never on words that happen to appear in the quotes FILE PATH the shared loader prefixes to every error

_UNRELATED_LOADER_ERRORS = [
    "{p}/sampled_at_local/quotes.json: 'quotes' must be an object keyed by symbol",
    "{p}/market_state/quotes.json: 'quotes' must be an object keyed by symbol",
    "{p}/sampled_at_local_market_state.json: unreadable",
]


@pytest.mark.parametrize("real_error", _UNRELATED_LOADER_ERRORS)
@pytest.mark.parametrize("dry_run", [False, True])
def test_an_unrelated_loader_error_whose_path_names_a_metadata_field_is_re_raised_not_reason_coded(
        tmp_path, monkeypatch, real_error, dry_run):
    def fake_load_quotes(path):
        raise ValueError(real_error.format(p=tmp_path))
    monkeypatch.setattr(driver.batch5, "load_quotes", fake_load_quotes)
    with pytest.raises(ValueError, match="must be an object keyed by symbol|unreadable"):
        _call(tmp_path, dry_run=dry_run, synthetic=dry_run)


def test_a_metadata_error_under_a_path_that_names_the_other_field_reports_only_the_field_that_failed(
        tmp_path, monkeypatch):
    def fake_load_quotes(path):
        raise ValueError(f"{tmp_path}/market_state/quotes.json: 'sampled_at_local' must be a non-empty string")
    monkeypatch.setattr(driver.batch5, "load_quotes", fake_load_quotes)
    result, _ledger = _call(tmp_path, dry_run=False, synthetic=False)
    assert result["reason_code"] == "quotes_not_market_hours"
    assert "sampled_at_local" in result["reason"] and "market_state" not in result["reason"]
    assert str(tmp_path) not in result["reason"]


# ------------------------------------------------------------------------------------------------
# The opt-in history end. Off by default; it only DROPS sessions dated after a
# chosen NYSE session, in memory, after the frames were loaded and cached; it fails closed; and it is
# recorded everywhere the driver records anything. The tests above are the oracle for the default.
# ------------------------------------------------------------------------------------------------

from mining import history_cutoff as hc  # noqa: E402

_HE_RUN = "gcr-he"
# The real-run guard refuses a history end that drops more than 10 sessions from any series unless it is told
# to accept more. The synthetic worlds below end 200 or more sessions before their data does, so a REAL run
# that must reach past the guard passes this explicitly (the guard itself is tested in its own section).
_HE_ACCEPT_ALL = 10 ** 6


def _iso(stamp):
    return pd.Timestamp(stamp).strftime("%Y-%m-%d")


def _he_run(tmp_path, history_end, *, world=None, dry_run=True, synthetic=None, run_id=_HE_RUN,
            tested_date="2026-09-24", accept_open_final_window=False, research_root=None,
            max_history_drop=None, quotes_overrides=None):
    """compute_and_record with a history end on a fresh fixture under tmp_path (its frozen futures are
    written from the same world): ``(result, ledger_path, research_root)``. ``max_history_drop`` is passed
    only when given (the driver's own default otherwise)."""
    world = world or _world()
    synthetic = dry_run if synthetic is None else synthetic
    catalog_path, quotes_path, ledger_path, default_root, futures_dir = _fixture(
        tmp_path, synthetic=synthetic, world=world, quotes_overrides=quotes_overrides)
    research_root = research_root or default_root
    extra = {} if max_history_drop is None else {"max_history_drop": max_history_drop}
    result = driver.compute_and_record(
        catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path, run_id=run_id,
        tested_date=tested_date, dry_run=dry_run, signal_source_path="scripts/run_gold_copper_overlay_abi1.py",
        signal_source_sha256="a" * 64, research_root=research_root, ohlc_loader=_loader(world),
        futures_dir=futures_dir, n_random_draws=6, accept_open_final_window=accept_open_final_window,
        history_end=history_end, **extra)
    return result, ledger_path, research_root


def _he_real(tmp_path, history_end, *, tested_date="2026-09-24", **kwargs):
    """A REAL (non-dry) run whose quote capture is an open-market sample dated ``tested_date`` (the
    market-hours guard needs the capture date and the tested date to agree)."""
    return _he_run(tmp_path, history_end, dry_run=False, synthetic=False, tested_date=tested_date,
                   quotes_overrides={"sampled_at_local": f"{tested_date}T10:00:00"}, **kwargs)


def _spy_on_scoring_inputs(monkeypatch):
    """Capture the last bar date of EVERY price frame handed to the curve builders, the evaluator, the
    descriptive statistics, the controls and the note, plus the account curve of every evaluation. A frame variable
    captured before the trim and used afterwards would show up here as a frame that ends after the cutoff."""
    seen = {"frames": [], "curves": []}

    def record(name, args, kwargs):
        for value in list(args) + list(kwargs.values()):
            if isinstance(value, pd.DataFrame) and isinstance(value.index, pd.DatetimeIndex) and len(value):
                seen["frames"].append((name, value.index.max()))

    def wrap(owner, attr):
        original = getattr(owner, attr)

        def spy(*args, **kwargs):
            record(attr, args, kwargs)
            return original(*args, **kwargs)
        monkeypatch.setattr(owner, attr, spy)

    for owner, attr in ((driver.mod, "prepare"), (driver.mod, "build_curve"), (driver.mod, "build_always_on_curve"),
                        (driver.mod, "curve_from_states"),
                        (driver.benchmark_relative, "cost_adjusted_benchmark_daily_returns"),
                        (driver, "overlay_diagnostics"), (driver, "control_arms"), (driver, "build_ledger_note")):
        wrap(owner, attr)
    original_evaluate = driver.abi1_driver.evaluate_family

    def spy_evaluate(curve, **kwargs):
        record("evaluate_family", (), kwargs)
        seen["curves"].append(curve["strategy_daily"].copy())
        return original_evaluate(curve, **kwargs)
    monkeypatch.setattr(driver.abi1_driver, "evaluate_family", spy_evaluate)
    return seen


def _poison_after(world, cutoff, seed):
    """The world with every value strictly after ``cutoff`` replaced by a wildly different but VALID one
    (SPY rescaled by a fresh random factor of up to 6 percent each day, several times its normal daily
    move, and the futures by large ones; the bill yield moved by several points). The poison stays a
    valid history, so a run that reads it does not refuse: it just scores differently."""
    rs = np.random.RandomState(seed)
    out = dict(world)
    for key in ("gc", "hg"):
        series = world[key].copy()
        after = series.index > cutoff
        series[after] = series[after] * np.exp(rs.normal(0.0, 1.0, int(after.sum())))
        out[key] = series
    spy = world["SPY"].copy()
    after = spy.index > cutoff
    columns = ["Open", "High", "Low", "Close"]
    factor = np.clip(np.exp(rs.normal(0.0, 0.03, (int(after.sum()), 1))), 0.94, 1.06)
    spy.loc[after, columns] = spy.loc[after, columns].to_numpy() * factor
    out["SPY"] = spy
    irx = world["^IRX"].copy()
    irx.loc[irx.index > cutoff, columns] += 7.0
    out["^IRX"] = irx
    return out


# ---- (a) off by default

def test_without_a_history_end_the_history_end_code_is_never_reached_and_the_result_carries_no_trace(
        tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the default path must not touch the history-end code")
    for name in ("parse_history_end", "trim_frames_to_history_end", "history_end_clause"):
        monkeypatch.setattr(driver.history_cutoff, name, refuse)
    world = _world()
    result, ledger_path, _ = _he_run(tmp_path, None, world=world)
    assert "reason_code" not in result
    for key in ("history_end", "raw_last_bar_dates"):
        assert key not in result
    assert hc.HISTORY_END_MARKER not in result["cost_model_note"]
    assert "history end" not in result["cost_model_note"].lower()
    assert result["last_bar_dates"]["SPY"] == _iso(world["SPY"].index[-1])
    assert not os.path.exists(ledger_path)


# ---- (b) a vendor hole that refuses today is scoreable when the history is ended before it

def test_an_interior_vendor_hole_that_refuses_by_default_is_scoreable_by_a_dry_run_ended_before_the_hole(tmp_path):
    world = _world()
    sessions = world["SPY"].index
    hole = sessions[-30]
    holed = dict(world)
    holed["SPY"] = world["SPY"].drop(index=hole)
    today, _ = _call(_mk(tmp_path, "default"), world=holed)                        # without the setting it refuses
    assert today["reason_code"] == "missing_exchange_session" and _iso(hole) in today["missing_dates"]
    cutoff = _iso(sessions[-31])
    result, ledger_path, _ = _he_run(_mk(tmp_path, "trimmed"), cutoff, world=holed)
    assert "reason_code" not in result and "verdict" in result
    assert result["account_end"] == cutoff and result["last_bar_dates"] == {"SPY": cutoff, "^IRX": cutoff}
    assert result["raw_last_bar_dates"] == {"SPY": _iso(sessions[-1]), "^IRX": _iso(sessions[-1])}
    provenance = result["history_end"]
    assert provenance["history_end"] == cutoff and provenance["opt_in"] is True
    assert provenance["dropped_dates"] == [_iso(d) for d in sessions[-30:]]         # the hole is dropped with the rest
    assert provenance["by_symbol"]["SPY"]["dropped_sessions"] == 29                   # SPY has no bar on the hole itself
    assert provenance["by_symbol"]["^IRX"]["dropped_sessions"] == 30
    assert not os.path.exists(ledger_path)


def test_a_hole_before_the_history_end_is_still_refused_on_the_trimmed_frames(tmp_path):
    world = _world()
    sessions = world["SPY"].index
    holed = dict(world)
    holed["SPY"] = world["SPY"].drop(index=sessions[-60])
    result, ledger_path, _ = _he_run(tmp_path, _iso(sessions[-30]), world=holed)
    assert result["reason_code"] == "missing_exchange_session" and _iso(sessions[-60]) in result["missing_dates"]
    assert "verdict" not in result and not os.path.exists(ledger_path)
    # this refusal comes AFTER the trim, so it still says the setting was on and what it did
    assert result["history_end_requested"] == _iso(sessions[-30])
    assert result["history_end"]["history_end"] == _iso(sessions[-30])


# ---- (c) the trimmed run is the prefix of the untrimmed run, and equals a run on data that stops there

def test_a_run_ended_at_a_history_end_equals_the_run_on_data_that_stops_there_and_is_a_prefix_of_the_full_run(
        tmp_path, monkeypatch):
    world = _world()
    cutoff = world["SPY"].index[-700]                 # far enough back that the trimmed run still evaluates a holdout
    seen = _spy_on_scoring_inputs(monkeypatch)
    trimmed, _, _ = _he_run(_mk(tmp_path, "trimmed"), _iso(cutoff), world=world)
    curve_trimmed = seen["curves"][0]
    seen["curves"].clear()
    stopped, _, _ = _he_run(_mk(tmp_path, "stopped"), None, world=_cut(world, cutoff))
    curve_stopped = seen["curves"][0]
    seen["curves"].clear()
    full, _, _ = _he_run(_mk(tmp_path, "full"), None, world=world)
    curve_full = seen["curves"][0]
    assert curve_trimmed.index[-1] == cutoff and curve_full.index[-1] == world["SPY"].index[-1]
    assert curve_trimmed.index.equals(curve_stopped.index)
    assert np.array_equal(curve_trimmed.to_numpy(), curve_stopped.to_numpy())
    for key in ("verdict", "stage", "dsr_value", "sharpe_daily_excess", "n_sessions", "n_trades", "account_start",
                "account_end", "holdout_psr", "time_in_market", "window"):
        assert trimmed[key] == stopped[key], key
    assert trimmed["overlay_diagnostics"]["windows"] == stopped["overlay_diagnostics"]["windows"]
    assert (trimmed["controls"]["random_window"]["draw_sharpes"]
            == stopped["controls"]["random_window"]["draw_sharpes"])
    # the prefix property: every session before the last one is what the untrimmed run computed (an open
    # window is closed at the last session, which pays its exit cost, so the last session is excluded)
    prefix = curve_trimmed.index[:-1]
    assert np.array_equal(curve_trimmed.loc[prefix].to_numpy(), curve_full.loc[prefix].to_numpy())
    assert len(prefix) > 500 and trimmed["account_end"] != full["account_end"]


# ---- (d) recorded in the note, the result and the provenance; absent when off

def test_the_history_end_is_recorded_in_the_note_the_result_and_the_provenance(tmp_path):
    world = _world()
    sessions = world["SPY"].index
    cutoff = _iso(sessions[-200])
    result, _, _ = _he_run(tmp_path, cutoff, world=world)
    provenance = result["history_end"]
    assert provenance["history_end"] == cutoff and provenance["decision"] == hc.DECISION
    assert sorted(provenance["by_symbol"]) == ["SPY", "^IRX"]
    for symbol in ("SPY", "^IRX"):
        record = provenance["by_symbol"][symbol]
        assert record["raw_last"] == _iso(sessions[-1]) and record["kept_last"] == cutoff
        assert record["dropped_sessions"] == 199 and len(record["dropped_dates"]) == 199
    assert provenance["total_dropped_series_sessions"] == 398 and len(provenance["dropped_dates"]) == 199
    note = result["cost_model_note"]
    assert note.count(hc.HISTORY_END_MARKER) == 1
    assert hc.history_end_clause(provenance) in note
    assert f"ended at {cutoff}" in note and f"SPY {_iso(sessions[-1])}" in note
    assert f"permanently ends at {cutoff}" in note and "cannot be re-scored on longer history" in note
    assert "The setting added, moved or repaired no bar" in note
    assert "not by data availability" not in note and "Nothing was filled or patched" not in note
    assert not _DOLLAR_FIGURE.search(note) and not _EIGHT_DIGITS.search(note)
    json.dumps(result, allow_nan=False, default=str)


def test_the_last_bar_sentence_says_later_vendor_bars_were_dropped_by_the_setting(tmp_path):
    world = _world()
    sessions = world["SPY"].index
    result, _, _ = _he_run(tmp_path, _iso(sessions[-200]), world=world)
    note = result["cost_model_note"]
    assert f"last SPY bar used is dated {_iso(sessions[-200])}" in note
    assert "not the vendor's final bar" in note and "dropped by the setting" in note
    assert f"raw last SPY bar is {_iso(sessions[-1])}" in note and "199 later session(s)" in note
    assert "last_spy_bar_may_be_live_partial=False" in note
    assert f"through {_iso(sessions[-200])}" in note                    # the price-data sentence describes the trimmed history


def _provenance_for(cutoff, raw_last, dropped):
    record = {"raw_first": "2000-01-03", "raw_last": raw_last, "kept_last": cutoff,
              "dropped_sessions": len(dropped), "dropped_dates": list(dropped)}
    return hc.build_provenance(pd.Timestamp(cutoff), {"SPY": record, "^IRX": dict(record)})


def test_the_last_bar_clause_without_a_history_end_is_the_ordinary_clause_and_with_one_names_the_raw_bar():
    plain = driver.last_bar_partial_clause("2026-09-21", "2026-09-24")
    assert plain == driver.last_bar_partial_clause("2026-09-21", "2026-09-24", history_end_provenance=None)
    assert "history end" not in plain[1]
    provenance = _provenance_for("2026-09-21", "2026-09-24", ["2026-09-22", "2026-09-23", "2026-09-24"])
    partial, text = driver.last_bar_partial_clause("2026-09-21", "2026-09-24", history_end_provenance=provenance)
    assert partial is False and text.startswith(plain[1])            # the ordinary sentence first, the setting's after
    assert "last_spy_bar_may_be_live_partial=False" in text
    assert "history end (2026-09-21)" in text and "not the vendor's final bar" in text
    assert "raw last SPY bar is 2026-09-24" in text and "3 later session(s) were dropped by the setting" in text
    # nothing dropped and the bar is dated the run date: the ordinary partial-bar warning still applies
    kept = _provenance_for("2026-09-24", "2026-09-24", [])
    partial, text = driver.last_bar_partial_clause("2026-09-24", "2026-09-24", history_end_provenance=kept)
    assert partial is True and "last_spy_bar_may_be_live_partial=True" in text
    assert "dropped no SPY bar" in text and "not the vendor's final bar" not in text


def test_a_real_run_records_the_history_end_in_the_ledger_note_and_pins_the_holdout_end_at_it(tmp_path):
    world = _world()
    cutoff = _iso(_low_end_world()["SPY"].index[-1])                    # no window is open there
    result, ledger_path, _ = _he_run(tmp_path, cutoff, world=world, dry_run=False, synthetic=False,
                                     max_history_drop=_HE_ACCEPT_ALL)
    assert "reason_code" not in result and result["stage"] in _SCORED
    assert result["account_end"] == cutoff and result["last_bar_dates"]["SPY"] == cutoff
    outcomes = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    assert len(outcomes) == 1
    note = outcomes[0]["note"]
    assert hc.history_end_clause(result["history_end"]) in note
    assert note.startswith(_RECORDED_PREFIX) and note.index(hc.HISTORY_END_MARKER) < note.index("cost model:")
    assert "permanently ends at " + cutoff in note
    pinned = trial_ledger.get_pinned_holdout_window(ledger_path, _FAMILY)
    assert pinned["holdout_end"] == cutoff
    assert not _DOLLAR_FIGURE.search(note) and not _EIGHT_DIGITS.search(note)


# ---- (e) refusals fail closed and write nothing

_HE_DOOR_REFUSALS = [
    ("2013-12-28", "history_end_not_a_session"),          # a Saturday
    ("2013-12-25", "history_end_not_a_session"),          # an exchange holiday
    ("2013-12", "history_end_invalid"), ("not-a-date", "history_end_invalid"), ("", "history_end_invalid"),
    ("2013-12-20T10:00:00", "history_end_invalid"), ("2013-02-30", "history_end_invalid"),
    (12345, "history_end_invalid"),
]


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("value,code", _HE_DOOR_REFUSALS)
def test_a_malformed_or_non_session_history_end_is_refused_at_the_door_and_nothing_is_written(
        tmp_path, value, code, dry_run):
    result, ledger_path, research_root = _he_run(tmp_path, value, dry_run=dry_run)
    assert result["reason_code"] == code and result["family_id"] == _FAMILY
    assert result["history_end_requested"] == str(value) and result["symbols"] == []
    assert "verdict" not in result and "history_end" not in result
    assert not os.path.exists(ledger_path)
    assert not os.path.exists(os.path.join(research_root, _HE_RUN))     # nothing fetched, nothing cached


@pytest.mark.parametrize("dry_run", [True, False])
def test_a_history_end_no_series_covers_is_refused_and_nothing_is_recorded(tmp_path, dry_run):
    result, ledger_path, _ = _he_run(tmp_path, "2014-01-02", dry_run=dry_run)     # a session after every series ends
    assert result["reason_code"] == "history_end_not_covered" and sorted(result["symbols"]) == ["SPY", "^IRX"]
    assert "verdict" not in result and "history_end" not in result and not os.path.exists(ledger_path)


@pytest.mark.parametrize("dry_run", [True, False])
def test_a_history_end_missing_from_one_series_names_that_series_and_nothing_is_recorded(tmp_path, dry_run):
    world = _world()
    cutoff = world["SPY"].index[-50]
    world["^IRX"] = world["^IRX"].drop(index=cutoff)
    result, ledger_path, _ = _he_run(tmp_path, _iso(cutoff), world=world, dry_run=dry_run)
    assert result["reason_code"] == "history_end_not_covered" and result["symbols"] == ["^IRX"]
    assert "^IRX" in result["reason"] and "verdict" not in result and not os.path.exists(ledger_path)


# ---- (f) the trim never reaches the OHLC cache

def test_a_history_end_never_reaches_the_ohlc_cache_and_a_later_real_run_still_refuses_the_stale_cache(tmp_path):
    world = _world()
    cutoff = _iso(world["SPY"].index[-200])
    result, _, research_root = _he_run(_mk(tmp_path, "dry"), cutoff, world=world, run_id="shared")
    assert "reason_code" not in result and result["ohlc_sources"] == {"SPY": "fetched", "^IRX": "fetched"}
    cache_dir = os.path.join(research_root, "shared", "artifacts", "ohlc")
    for symbol in ("SPY", "^IRX"):                                       # the full vendor history is still cached
        cached = pd.read_csv(os.path.join(cache_dir, driver.batch5._cache_name(symbol)))
        assert len(cached) == len(world[symbol]) and cached["Date"].iloc[-1] == _iso(world[symbol].index[-1])
    again, _, _ = _he_run(_mk(tmp_path, "again"), cutoff, world=world, run_id="shared", research_root=research_root)
    assert again["ohlc_sources"] == {"SPY": "cache", "^IRX": "cache"} and again["account_end"] == cutoff
    for real_history_end in (None, cutoff):
        refused, ledger_path, _ = _he_run(_mk(tmp_path, f"real_{real_history_end}"), real_history_end, world=world,
                                          dry_run=False, synthetic=False, run_id="shared", research_root=research_root)
        assert refused["reason_code"] == "stale_ohlc_cache" and sorted(refused["cached_symbols"]) == ["SPY", "^IRX"]
        assert "history_end" not in refused and not os.path.exists(ledger_path)


# ---- (g) nothing after the history end reaches the score

def test_every_frame_the_scoring_path_receives_ends_at_the_history_end(tmp_path, monkeypatch):
    world = _world()
    cutoff = world["SPY"].index[-200]
    seen = _spy_on_scoring_inputs(monkeypatch)
    _he_run(_mk(tmp_path, "trimmed"), _iso(cutoff), world=world)
    names = {name for name, _ in seen["frames"]}
    assert {"prepare", "build_curve", "build_always_on_curve", "curve_from_states", "evaluate_family",
            "cost_adjusted_benchmark_daily_returns", "overlay_diagnostics", "control_arms"} <= names
    assert all(last <= cutoff for _, last in seen["frames"]), [x for x in seen["frames"] if x[1] > cutoff]
    seen["frames"].clear()
    _he_run(_mk(tmp_path, "untrimmed"), None, world=world)          # the spy sees a late frame when nothing is trimmed
    assert any(last > cutoff for _, last in seen["frames"])


def test_nothing_after_the_history_end_reaches_the_score(tmp_path):
    world = _world()
    cutoff = world["SPY"].index[-1000]      # the poisoned tail is 1000 sessions: in-sample AND the whole holdout
    poisoned = _poison_after(world, cutoff, seed=3)
    plain_a, _, _ = _he_run(_mk(tmp_path, "plain_a"), None, world=world)
    plain_b, _, _ = _he_run(_mk(tmp_path, "plain_b"), None, world=poisoned)
    assert plain_a["sharpe_daily_excess"] != plain_b["sharpe_daily_excess"]     # the poison is real
    assert plain_a["holdout_psr"] != plain_b["holdout_psr"]
    a, _, _ = _he_run(_mk(tmp_path, "a"), _iso(cutoff), world=world)
    b, _, _ = _he_run(_mk(tmp_path, "b"), _iso(cutoff), world=poisoned)
    assert "reason_code" not in a and "reason_code" not in b and a["holdout_evaluated"] is True

    def comparable(result):              # the frozen futures files differ (poisoned after the cutoff), nothing else
        return json.dumps({k: v for k, v in result.items() if k != "futures_provenance"}, sort_keys=True, default=str)
    assert comparable(a) == comparable(b)


# ---- (h) the final-window rule still runs on the trimmed frames

def test_a_real_run_ended_inside_an_open_window_still_refuses_the_truncated_final_window_unless_accepted(tmp_path):
    cut_world, _ = _open_window_world(3)
    cutoff = _iso(cut_world["SPY"].index[-1])
    world = _world()
    refused, ledger_path, _ = _he_run(_mk(tmp_path, "refused"), cutoff, world=world, dry_run=False, synthetic=False,
                                      max_history_drop=_HE_ACCEPT_ALL)
    assert refused["reason_code"] == "truncated_final_window" and "--accept-open-final-window" in refused["reason"]
    assert "verdict" not in refused and not os.path.exists(ledger_path)
    dry, _, _ = _he_run(_mk(tmp_path, "dry"), cutoff, world=world)
    assert dry["overlay_diagnostics"]["final_window_truncated"] is True and dry["account_end"] == cutoff
    accepted, ledger_path, _ = _he_run(_mk(tmp_path, "accepted"), cutoff, world=world, dry_run=False,
                                       synthetic=False, accept_open_final_window=True,
                                       max_history_drop=_HE_ACCEPT_ALL)
    assert accepted["stage"] in _SCORED + ("insufficient_evidence",) and accepted["accepted_open_final_window"] is True
    outcomes = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    note = outcomes[0]["note"]
    assert "final_window_truncated=True" in note and "accepted_open_final_window=True" in note
    assert hc.history_end_clause(accepted["history_end"]) in note


# ---- the futures are not trimmed and the note says why that is safe

_FUTURES_LOADED_WHOLE = "loaded whole and validated"
_FUTURES_ONE_ROW_PER_DECISION = "each decision reads only the last futures row on or before its own read date"
_FUTURES_NO_LATER_DECISION = "no decision is later than the history end (the cutoff)"
_FUTURES_CANNOT_INFLUENCE = "no futures row dated after the cutoff can influence a decision"


def test_the_note_states_what_the_futures_reads_can_and_cannot_see_under_a_history_end(tmp_path):
    # the frozen futures files are loaded whole and validated (so a malformed later row could still refuse the
    # run): the note must not claim that no later row is "read", only that none can influence a decision
    world = _world()
    result, _, _ = _he_run(tmp_path, _iso(world["SPY"].index[-200]), world=world)
    note = result["cost_model_note"]
    for sentence in (_FUTURES_LOADED_WHOLE, _FUTURES_ONE_ROW_PER_DECISION, _FUTURES_NO_LATER_DECISION,
                     _FUTURES_CANNOT_INFLUENCE):
        assert sentence in note, sentence
    assert "dated after the history end is read" not in note and "row dated after the history end is read" not in note
    default, _, _ = _he_run(_mk(tmp_path, "default"), None, world=world)
    for sentence in (_FUTURES_LOADED_WHOLE, "can influence a decision", "history end and the futures"):
        assert sentence not in default["cost_model_note"], sentence


# ---- the CLI

def _cli_history_end(tmp_path, monkeypatch, world, value):
    catalog_path, quotes_path, ledger_path, research_root, futures_dir = _fixture(tmp_path, world=world)
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(world))
    code = driver.main(["--dry-run", "--quotes", quotes_path, "--catalog", catalog_path, "--ledger", ledger_path,
                        "--research-root", research_root, "--futures-dir", futures_dir, "--random-draws", "3",
                        "--history-end", value])
    return code, ledger_path, research_root


def test_cli_history_end_is_threaded_and_written_to_the_result_file(tmp_path, capsys, monkeypatch):
    world = _world()
    cutoff = _iso(world["SPY"].index[-200])
    code, ledger_path, research_root = _cli_history_end(tmp_path, monkeypatch, world, cutoff)
    assert code == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["output"].endswith("gold-copper-overlay-dry-run.json")
    with open(os.path.join(research_root, driver.DEFAULT_RUN_ID, "gold-copper-overlay-dry-run.json"),
              encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["history_end"]["history_end"] == cutoff
    assert payload["result"]["history_end"] == payload["history_end"]
    assert payload["result"]["account_end"] == cutoff
    assert hc.HISTORY_END_MARKER in payload["result"]["cost_model_note"]
    assert printed["result"]["raw_last_bar_dates"]["SPY"] == _iso(world["SPY"].index[-1])
    assert not os.path.exists(ledger_path)


def test_cli_without_history_end_writes_no_history_end_key(tmp_path, capsys, monkeypatch):
    world = _world()
    catalog_path, quotes_path, ledger_path, research_root, futures_dir = _fixture(tmp_path, world=world)
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(world))
    assert driver.main(["--dry-run", "--quotes", quotes_path, "--catalog", catalog_path, "--ledger", ledger_path,
                        "--research-root", research_root, "--futures-dir", futures_dir, "--random-draws", "3"]) == 0
    with open(os.path.join(research_root, driver.DEFAULT_RUN_ID, "gold-copper-overlay-dry-run.json"),
              encoding="utf-8") as fh:
        payload = json.load(fh)
    assert "history_end" not in payload and "history_end" not in payload["result"]


@pytest.mark.parametrize("value,code", [("2013-12-28", "history_end_not_a_session"),
                                        ("not-a-date", "history_end_invalid"), ("", "history_end_invalid"),
                                        ("2014-01-02", "history_end_not_covered")])
def test_cli_refused_history_end_writes_no_result_file_and_exits_nonzero(tmp_path, capsys, monkeypatch, value, code):
    code_returned, ledger_path, research_root = _cli_history_end(tmp_path, monkeypatch, _world(), value)
    assert code_returned == 2
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["output"] is None and printed["result"]["reason_code"] == code
    assert not os.path.exists(os.path.join(research_root, driver.DEFAULT_RUN_ID, "gold-copper-overlay-dry-run.json"))
    assert not os.path.exists(os.path.join(research_root, driver.DEFAULT_RUN_ID, "gold-copper-overlay-results.json"))
    assert not os.path.exists(ledger_path)


def test_the_driver_docstring_describes_the_history_end_exception_to_never_trimmed():
    text = " ".join(driver.__doc__.split())
    assert "--history-end" in text and "opt-in history-end setting" in text
    for code in ("history_end_invalid", "history_end_not_a_session", "history_end_not_covered"):
        assert code in text
    assert "permanent" in text.lower() and "after the frames were loaded and cached" in text.lower()


def test_a_date_or_midnight_timestamp_history_end_is_read_like_the_string_and_a_datetime_with_a_time_is_refused(
        tmp_path):
    world = _world()
    cutoff = world["SPY"].index[-200]
    by_string, _, _ = _he_run(_mk(tmp_path, "s"), _iso(cutoff), world=world)
    for name, value in (("d", cutoff.date()), ("t", cutoff)):
        result, _, _ = _he_run(_mk(tmp_path, name), value, world=world)
        assert result["history_end"] == by_string["history_end"] and result["account_end"] == by_string["account_end"]
    refused, ledger_path, _ = _he_run(_mk(tmp_path, "time"), cutoff + pd.Timedelta(hours=9), world=world)
    assert refused["reason_code"] == "history_end_invalid" and not os.path.exists(ledger_path)


# ---- (f) an UNSCORED run spends the family's permanent ABI-1 slot too, so under a history end the shared
# recorder is handed the same clause for the note of that row (its ``unscored_note`` keyword)

_HE_UNSCORED_CUTOFF = "2005-09-30"      # 45 account sessions on the synthetic world: not_scoreable
_RECORDER_KEYWORDS_DEFAULT = {"catalog_entry", "instrument_set", "cash_convention", "benchmark", "tax_policy",
                            "ticker_label", "extra_note", "signal_source_path", "signal_source_sha256"}


def _spy_on_the_recorder(monkeypatch):
    """The keyword arguments of every call the driver makes to the shared ledger writer."""
    calls = []
    original = driver.abi1_driver.record_abi1_outcome

    def spy(*args, **kwargs):
        calls.append(dict(kwargs))
        return original(*args, **kwargs)
    monkeypatch.setattr(driver.abi1_driver, "record_abi1_outcome", spy)
    return calls


def _world_and_cutoff_for(kind):
    """``(world, history_end)`` whose real run ends UNSCORED (``unscored``) or SCORED (``scored``)."""
    world = _world()
    if kind == "unscored":
        return world, _HE_UNSCORED_CUTOFF
    return world, _iso(_low_end_world()["SPY"].index[-1])            # no window is open there


@pytest.mark.parametrize("kind", ["unscored", "scored"])
def test_under_a_history_end_the_recorder_is_handed_the_history_end_clause_as_the_unscored_note(
        tmp_path, monkeypatch, kind):
    world, cutoff = _world_and_cutoff_for(kind)
    calls = _spy_on_the_recorder(monkeypatch)
    result, _, _ = _he_run(tmp_path, cutoff, world=world, dry_run=False, synthetic=False,
                           max_history_drop=_HE_ACCEPT_ALL)
    assert (result["stage"] == "insufficient_evidence") == (kind == "unscored")
    assert len(calls) == 1
    clause = hc.history_end_clause(result["history_end"])
    assert hc.HISTORY_END_MARKER in clause and cutoff in clause
    assert calls[0]["unscored_note"] == clause                      # verbatim: the same clause the note leads with
    assert clause in calls[0]["extra_note"]
    assert set(calls[0]) == _RECORDER_KEYWORDS_DEFAULT | {"unscored_note"}


@pytest.mark.parametrize("kind", ["unscored", "scored"])
def test_without_a_history_end_the_recorder_call_carries_only_the_default_keywords(tmp_path, monkeypatch, kind):
    world, _ = _world_and_cutoff_for(kind)
    world = _cut(world, _HE_UNSCORED_CUTOFF) if kind == "unscored" else _low_end_world()
    calls = _spy_on_the_recorder(monkeypatch)
    result, _ = _call(tmp_path, dry_run=False, synthetic=False, world=world)
    assert (result["stage"] == "insufficient_evidence") == (kind == "unscored") and "history_end" not in result
    assert len(calls) == 1
    assert set(calls[0]) == _RECORDER_KEYWORDS_DEFAULT                # the keyword is not passed at all, not even as None
    assert hc.HISTORY_END_MARKER not in calls[0]["extra_note"]


def test_a_dry_run_under_a_history_end_never_calls_the_recorder(tmp_path, monkeypatch):
    calls = _spy_on_the_recorder(monkeypatch)
    world, cutoff = _world_and_cutoff_for("unscored")
    result, ledger_path, _ = _he_run(tmp_path, cutoff, world=world, dry_run=True)
    assert result["stage"] == "insufficient_evidence" and calls == [] and not os.path.exists(ledger_path)


def test_a_real_unscored_run_under_a_history_end_writes_the_clause_as_the_note_of_its_ledger_row(tmp_path):
    """The real compute path, a temporary ledger: the synthetic world ended at the cutoff is not scoreable,
    so the permanent row is an insufficient-evidence row, and it states the setting."""
    world, cutoff = _world_and_cutoff_for("unscored")
    result, ledger_path, _ = _he_run(tmp_path, cutoff, world=world, dry_run=False, synthetic=False,
                                     max_history_drop=_HE_ACCEPT_ALL)
    assert result["verdict"] == "not_scoreable" and result["stage"] == "insufficient_evidence"
    assert result["account_end"] == cutoff and result["last_bar_dates"]["SPY"] == cutoff
    assert result["history_end"]["total_dropped_series_sessions"] > 0
    outcomes = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    assert len(outcomes) == 1
    row = outcomes[0]
    assert row["stage"] == "insufficient_evidence" and row["construction"] == "ABI-1"
    note = row["note"]
    assert note == hc.history_end_clause(result["history_end"])
    assert hc.HISTORY_END_MARKER in note and f"ended at {cutoff}" in note
    assert f"permanently ends at {cutoff}" in note and "cannot be re-scored on longer history" in note
    assert "cost model:" not in note                                 # only the clause, never the scored-run note
    assert not _DOLLAR_FIGURE.search(note) and not _EIGHT_DIGITS.search(note)
    assert row["evidence"]["observed"]["instrument_set"] == ["SPY"]  # the unscored evidence payload is unchanged


def test_a_real_unscored_run_without_a_history_end_still_writes_no_note(tmp_path):
    """The default: without a history end the unscored row carries no note."""
    world = _cut(_world(), _HE_UNSCORED_CUTOFF)
    result, ledger_path = _call(tmp_path, dry_run=False, synthetic=False, world=world)
    assert result["stage"] == "insufficient_evidence"
    outcomes = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    assert len(outcomes) == 1 and outcomes[0]["note"] is None


def test_the_driver_docstring_says_the_history_end_clause_reaches_the_note_of_an_unscored_row_too():
    text = " ".join(driver.__doc__.split())
    assert "unscored_note" in text and "not_scoreable" in text and "holdout_not_evaluable" in text
    assert "insufficient_evidence" in text and "verbatim" in text.lower()


# ------------------------------------------------------------------------------------------------
# The real-run guard (--max-history-drop), ONE standard refusal shape, refusals that
# come after the trim, the CLI traps, the frozen futures' last bars and the doc truth.
# ------------------------------------------------------------------------------------------------

_HE_REFUSAL_KEYS = ["reason_code", "reason", "history_end_requested", "symbols", "family_id"]
_HE_CACHE_SENTENCE = "retry with a fresh --run-id"
_HE_HOW_TO_ACCEPT = "pass --max-history-drop N to accept a larger drop"


def _real_cli(tmp_path, monkeypatch, world, *extra, tested_date="2026-09-24", run_id="gcr-cli"):
    """The CLI for a REAL run against a fresh fixture (real-looking quotes dated ``tested_date``, the OHLC loader
    monkeypatched to ``world``): ``(exit_code, ledger_path, research_root)``."""
    catalog_path, quotes_path, ledger_path, research_root, futures_dir = _fixture(
        tmp_path, synthetic=False, world=world, quotes_overrides={"sampled_at_local": f"{tested_date}T10:00:00"})
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(world))
    code = driver.main(["--quotes", quotes_path, "--catalog", catalog_path, "--ledger", ledger_path,
                        "--research-root", research_root, "--futures-dir", futures_dir,
                        "--run-id", run_id, "--tested-date", tested_date, *extra])
    return code, ledger_path, research_root


def _worst_drop(provenance):
    return max(rec["dropped_sessions"] for rec in provenance["by_symbol"].values())


def _cache_dir(research_root, run_id=_HE_RUN):
    return os.path.join(research_root, run_id, "artifacts", "ohlc")


# ---- (a) the real-run guard

def test_the_compute_function_defaults_to_the_shared_drop_limit():
    import inspect
    default = inspect.signature(driver.compute_and_record).parameters["max_history_drop"].default
    assert default == hc.MAX_UNACCEPTED_DROP_SESSIONS == 10


def test_a_real_run_refuses_a_history_end_later_than_the_tested_date_and_records_nothing(tmp_path):
    world = _world()
    sessions = world["SPY"].index
    cutoff, tested = _iso(sessions[-200]), _iso(sessions[-201])            # the tested date is the session before
    result, ledger_path, research_root = _he_real(tmp_path, cutoff, world=world, tested_date=tested,
                                                  max_history_drop=_HE_ACCEPT_ALL)
    assert result["reason_code"] == "history_end_after_tested_date"
    assert cutoff in result["reason"] and tested in result["reason"]
    assert result["symbols"] == [] and result["history_end_requested"] == cutoff
    assert "verdict" not in result and not os.path.exists(ledger_path)
    # the guard runs after the load step, so the per-run OHLC cache is kept (a real retry needs a fresh run id)
    assert os.path.isdir(_cache_dir(research_root))


def test_a_real_run_whose_history_end_equals_the_tested_date_is_not_refused_by_the_guard(tmp_path):
    world = _world()
    cutoff = _iso(_low_end_world()["SPY"].index[-1])
    result, ledger_path, _ = _he_real(tmp_path, cutoff, world=world, tested_date=cutoff,
                                      max_history_drop=_HE_ACCEPT_ALL)
    assert "reason_code" not in result and result["stage"] in _SCORED and result["history_end_requested"] == cutoff
    assert len([r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]) == 1


def test_a_real_run_refuses_a_history_end_that_drops_more_than_the_default_limit_and_records_nothing(tmp_path):
    world = _world()
    cutoff = _iso(world["SPY"].index[-200])                                  # 199 sessions lost by each series
    result, ledger_path, research_root = _he_real(tmp_path, cutoff, world=world)     # the driver's own default limit
    assert result["reason_code"] == "history_end_drop_too_large" and result["symbols"] == ["SPY", "^IRX"]
    assert "199" in result["reason"] and f"more than the {hc.MAX_UNACCEPTED_DROP_SESSIONS}" in result["reason"]
    assert _HE_HOW_TO_ACCEPT in result["reason"] and result["history_end_requested"] == cutoff
    assert "verdict" not in result and not os.path.exists(ledger_path)
    assert os.path.isdir(_cache_dir(research_root))


def test_an_explicit_max_history_drop_accepts_exactly_that_drop_and_one_less_refuses(tmp_path):
    world = _world()
    cutoff = _iso(_low_end_world()["SPY"].index[-1])
    dry, _, _ = _he_run(_mk(tmp_path, "dry"), cutoff, world=world)
    worst = _worst_drop(dry["history_end"])
    assert worst > hc.MAX_UNACCEPTED_DROP_SESSIONS
    refused, ledger_path, _ = _he_real(_mk(tmp_path, "one_less"), cutoff, world=world, max_history_drop=worst - 1)
    assert refused["reason_code"] == "history_end_drop_too_large" and str(worst) in refused["reason"]
    assert not os.path.exists(ledger_path)
    accepted, ledger_path, _ = _he_real(_mk(tmp_path, "exact"), cutoff, world=world, max_history_drop=worst)
    assert "reason_code" not in accepted and accepted["stage"] in _SCORED
    assert accepted["history_end_requested"] == cutoff and "history_end_guard_would_refuse" not in accepted
    assert len([r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]) == 1


def _assert_door_refusal(result, ledger_path, research_root, cutoff, *, mentions):
    """The standard door refusal of a history end whose guard arguments are unusable: the standard keys,
    history_end_invalid, no cache-kept sentence (nothing was loaded, so no cache exists to keep), no ledger row,
    and no per-run directory at all (nothing fetched, nothing cached)."""
    assert result["reason_code"] == "history_end_invalid" and mentions in result["reason"]
    assert list(result) == _HE_REFUSAL_KEYS
    assert result["history_end_requested"] == cutoff and result["symbols"] == []
    assert _HE_CACHE_SENTENCE not in result["reason"] and "verdict" not in result
    assert not os.path.exists(ledger_path)
    assert not os.path.exists(_cache_dir(research_root))
    assert not os.path.exists(os.path.join(research_root, _HE_RUN))


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("bad", [-1, True, 2.5, "5"])
def test_an_unusable_max_history_drop_is_refused_at_the_door_by_real_and_dry_runs_alike(tmp_path, bad, dry_run):
    # a dry run rehearses the real run, so it refuses the unusable limit the same way, before anything is loaded
    world = _world()
    cutoff = _iso(world["SPY"].index[-200])
    result, ledger_path, research_root = _he_run(tmp_path, cutoff, world=world, dry_run=dry_run, max_history_drop=bad)
    _assert_door_refusal(result, ledger_path, research_root, cutoff, mentions="max_drop")


# a string no pandas parse rejects ("2026-9-25" is a real date to pandas) but the strict YYYY-MM-DD rule does
_BAD_TESTED_DATES = ["2026-9-25", "not-a-date", "", None, "2026-09-25T10:00:00", "2026-02-30", 2026_0925]


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("bad", _BAD_TESTED_DATES,
                         ids=["unpadded", "word", "empty", "none", "with_time", "impossible_day", "integer"])
def test_under_a_history_end_a_malformed_tested_date_is_a_door_refusal_not_an_escaping_exception(
        tmp_path, bad, dry_run):
    world = _world()
    cutoff = _iso(world["SPY"].index[-200])
    result, ledger_path, research_root = _he_run(tmp_path, cutoff, world=world, dry_run=dry_run, tested_date=bad)
    _assert_door_refusal(result, ledger_path, research_root, cutoff, mentions="tested_date")


def test_under_a_history_end_the_door_checks_the_guard_arguments_before_the_catalog_or_the_quotes(tmp_path):
    # neither the catalog nor the quotes file exists: reaching either would fail differently
    result = driver.compute_and_record(
        catalog_path=str(tmp_path / "missing.json"), quotes_path=str(tmp_path / "missing-quotes.json"),
        ledger_path=str(tmp_path / "ledger.jsonl"), run_id="x", tested_date="2026-9-25", dry_run=True,
        signal_source_path="p", signal_source_sha256="s", research_root=str(tmp_path / "research"),
        history_end="2013-12-20")
    assert result["reason_code"] == "history_end_invalid" and "tested_date" in result["reason"]
    assert result["history_end_requested"] == "2013-12-20"
    assert not (tmp_path / "research").exists() and not (tmp_path / "ledger.jsonl").exists()


def test_without_a_history_end_the_guard_arguments_are_not_checked_at_the_door(tmp_path):
    world = _world()
    # a limit nobody reads is ignored (even an unusable one)
    ignored, _, _ = _he_run(_mk(tmp_path, "limit"), None, world=world, max_history_drop="junk")
    assert "reason_code" not in ignored and "verdict" in ignored
    # a tested date pandas reads but the strict rule does not: accepted, no refusal
    lenient, _ = _call(_mk(tmp_path, "date"), world=world, tested_date="2026-9-25")
    assert "reason_code" not in lenient and "run tested date is 2026-09-25" in lenient["cost_model_note"]
    # a tested date pandas cannot read raises a plain ValueError, never a refusal payload
    with pytest.raises(ValueError, match="tested_date"):
        _call(_mk(tmp_path, "bad"), world=world, tested_date="not-a-date")


def test_a_dry_run_never_refuses_on_the_guard_and_says_what_the_real_run_would_do(tmp_path):
    world = _world()
    cutoff = _iso(_low_end_world()["SPY"].index[-1])
    big, _, _ = _he_run(_mk(tmp_path, "big"), cutoff, world=world)                 # the default limit
    assert "reason_code" not in big and "verdict" in big
    message = big["history_end_guard_would_refuse"]
    assert isinstance(message, str) and "would drop" in message and _HE_HOW_TO_ACCEPT in message
    real, _, _ = _he_real(_mk(tmp_path, "real"), cutoff, world=world)              # the same defaults, for real
    # the real reason is the same words plus the cache-kept sentence, so the rehearsal predicts it
    assert real["reason_code"] == "history_end_drop_too_large" and real["reason"].startswith(message)
    assert _HE_CACHE_SENTENCE in real["reason"]
    fine, _, _ = _he_run(_mk(tmp_path, "fine"), cutoff, world=world, max_history_drop=_HE_ACCEPT_ALL)
    assert "reason_code" not in fine and fine["history_end_guard_would_refuse"] is None
    early = (pd.Timestamp(cutoff) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")     # a tested date before the cutoff
    late, _, _ = _he_run(_mk(tmp_path, "late"), cutoff, world=world, tested_date=early, max_history_drop=_HE_ACCEPT_ALL)
    assert "reason_code" not in late and "verdict" in late
    assert "later than the run's tested date" in late["history_end_guard_would_refuse"]
    real_late, _, _ = _he_real(_mk(tmp_path, "real_late"), cutoff, world=world, tested_date=early,
                               max_history_drop=_HE_ACCEPT_ALL)
    assert real_late["reason_code"] == "history_end_after_tested_date"
    assert real_late["reason"].startswith(late["history_end_guard_would_refuse"])


def test_without_a_history_end_the_result_has_none_of_the_history_end_keys(tmp_path):
    world = _world()
    dry, _, _ = _he_run(tmp_path, None, world=world, max_history_drop=3)          # a limit without a setting is ignored
    assert "reason_code" not in dry
    for key in ("history_end", "history_end_requested", "history_end_guard_would_refuse", "raw_last_bar_dates"):
        assert key not in dry
    assert set(dry["futures_provenance"]) == {"through", "files", "sha256_verified", "directory"}
    assert "futures cache was not trimmed" not in dry["cost_model_note"]


# ---- (b) one refusal shape

@pytest.mark.parametrize("value,code", _HE_DOOR_REFUSALS)
def test_a_door_refusal_is_the_shared_standard_payload_without_the_cache_sentence(tmp_path, value, code):
    result, _, _ = _he_run(tmp_path, value)
    with pytest.raises(hc.HistoryCutoffError) as excinfo:
        hc.parse_history_end(value)
    assert list(result) == _HE_REFUSAL_KEYS
    assert result == hc.refusal_payload(excinfo.value, history_end_requested=value, family_id=_FAMILY)
    assert result["reason_code"] == code and _HE_CACHE_SENTENCE not in result["reason"]


@pytest.mark.parametrize("dry_run", [True, False])
def test_a_not_covered_refusal_has_the_standard_shape_and_says_the_cache_is_kept(tmp_path, dry_run):
    result, ledger_path, research_root = _he_run(tmp_path, "2014-01-02", dry_run=dry_run)
    assert list(result) == _HE_REFUSAL_KEYS and result["reason_code"] == "history_end_not_covered"
    assert sorted(result["symbols"]) == ["SPY", "^IRX"] and result["history_end_requested"] == "2014-01-02"
    assert _HE_CACHE_SENTENCE in result["reason"] and "is kept" in result["reason"]
    assert os.path.isdir(_cache_dir(research_root)) and not os.path.exists(ledger_path)


def test_the_guard_refusals_have_the_standard_shape_and_the_cache_sentence(tmp_path):
    """Both guard refusals are raised after the load step wrote the per-run OHLC
    cache, so (like not-covered) they say it is kept and a retry needs a fresh
    run-id; only the door refusals, raised before any load, do not."""
    world = _world()
    cutoff = _iso(world["SPY"].index[-200])
    big, _, _ = _he_real(_mk(tmp_path, "big"), cutoff, world=world)
    late, _, _ = _he_real(_mk(tmp_path, "late"), cutoff, world=world, tested_date=_iso(world["SPY"].index[-201]),
                          max_history_drop=_HE_ACCEPT_ALL)
    for refused, code in ((big, "history_end_drop_too_large"), (late, "history_end_after_tested_date")):
        assert refused["reason_code"] == code and list(refused) == _HE_REFUSAL_KEYS
        assert refused["family_id"] == _FAMILY and refused["history_end_requested"] == cutoff
        assert _HE_CACHE_SENTENCE in refused["reason"] and "is kept" in refused["reason"]


# ---- (i) a refusal that comes AFTER the trim still says the setting was on

def test_a_truncated_final_window_refusal_says_the_setting_ended_the_history_and_carries_its_provenance(tmp_path):
    cut_world, _ = _open_window_world(3)
    cutoff = _iso(cut_world["SPY"].index[-1])
    refused, ledger_path, _ = _he_real(_mk(tmp_path, "he"), cutoff, world=_world(), max_history_drop=_HE_ACCEPT_ALL)
    assert refused["reason_code"] == "truncated_final_window" and not os.path.exists(ledger_path)
    assert refused["history_end_requested"] == cutoff
    assert refused["history_end"]["history_end"] == cutoff and refused["history_end"]["opt_in"] is True
    assert f"ended at {cutoff}" in refused["reason"] and "--history-end" in refused["reason"]
    assert "--accept-open-final-window" in refused["reason"]
    plain, _ = _call(_mk(tmp_path, "plain"), dry_run=False, synthetic=False, world=cut_world)
    assert plain["reason_code"] == "truncated_final_window"                     # the same refusal without the setting
    assert "--history-end" not in plain["reason"]
    assert "history_end" not in plain and "history_end_requested" not in plain


def test_a_futures_refusal_after_the_trim_carries_the_history_end_requested_and_the_provenance(tmp_path):
    world = _world()
    short = {**world, "gc": world["gc"].loc[:"2013-10-31"], "hg": world["hg"].loc[:"2013-10-31"]}
    cutoff = _iso(world["SPY"].index[-20])                       # the November reads lie after the cached futures end
    result, ledger_path, _ = _he_run(tmp_path, cutoff, world=short, max_history_drop=_HE_ACCEPT_ALL)
    assert result["reason_code"] == "futures_read_hole" and "verdict" not in result
    assert result["history_end_requested"] == cutoff and result["history_end"]["history_end"] == cutoff
    assert not os.path.exists(ledger_path)


def test_a_calendar_refusal_after_the_trim_says_the_setting_was_on(tmp_path):
    world = _world()
    sessions = world["SPY"].index
    holed = dict(world)
    holed["SPY"] = world["SPY"].drop(index=sessions[-60])
    result, _, _ = _he_run(tmp_path, _iso(sessions[-30]), world=holed, max_history_drop=_HE_ACCEPT_ALL)
    assert result["reason_code"] == "missing_exchange_session"
    assert result["history_end_requested"] == _iso(sessions[-30]) and "history_end" in result


# ---- the frozen futures: report their last bars, say they were not trimmed, and say no later row can influence a decision

def test_under_a_history_end_the_result_and_note_report_each_futures_last_bar_and_that_it_was_not_trimmed(tmp_path):
    world = _world()
    world["hg"] = world["hg"].loc[:"2013-12-27"]                   # the two frozen series end on different days
    cutoff = _iso(world["SPY"].index[-200])
    result, _, _ = _he_run(tmp_path, cutoff, world=world)
    gc_last, hg_last = _iso(world["gc"].index[-1]), _iso(world["hg"].index[-1])
    assert gc_last != hg_last and min(gc_last, hg_last) > cutoff    # both cached series run past the cutoff, untrimmed
    futures = result["futures_provenance"]
    assert futures["last_bar_dates"] == {"GC=F": gc_last, "HG=F": hg_last}
    assert futures["trimmed_by_history_end"] is False
    assert _FUTURES_LOADED_WHOLE in futures["history_end_note"]
    assert _FUTURES_ONE_ROW_PER_DECISION in futures["history_end_note"]
    assert "is read" not in futures["history_end_note"] and "read only causally" not in futures["history_end_note"]
    note = result["cost_model_note"]
    assert f"GC=F {gc_last}" in note and f"HG=F {hg_last}" in note
    assert "not trimmed" in note and _FUTURES_CANNOT_INFLUENCE in note
    assert not _DOLLAR_FIGURE.search(note) and not _EIGHT_DIGITS.search(note)


def test_the_futures_last_bar_dates_do_not_move_with_the_history_end(tmp_path):
    world = _world()
    a, _, _ = _he_run(_mk(tmp_path, "a"), _iso(world["SPY"].index[-200]), world=world)
    b, _, _ = _he_run(_mk(tmp_path, "b"), _iso(world["SPY"].index[-400]), world=world)
    assert a["futures_provenance"]["last_bar_dates"] == b["futures_provenance"]["last_bar_dates"]
    assert a["futures_provenance"]["last_bar_dates"] == {"GC=F": _iso(world["gc"].index[-1]),
                                                         "HG=F": _iso(world["hg"].index[-1])}


# ---- the CLI

def test_cli_passes_the_success_payload_the_setting_and_the_guard_field_on_a_dry_run(tmp_path, capsys, monkeypatch):
    world = _world()
    cutoff = _iso(world["SPY"].index[-200])
    code, _, research_root = _cli_history_end(tmp_path, monkeypatch, world, cutoff)
    assert code == 0
    with open(os.path.join(research_root, driver.DEFAULT_RUN_ID, "gold-copper-overlay-dry-run.json"),
              encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["history_end_requested"] == cutoff and payload["result"]["history_end_requested"] == cutoff
    assert payload["result"]["history_end"] == payload["history_end"]
    assert "more than the 10" in payload["result"]["history_end_guard_would_refuse"]     # the default limit, threaded


def test_cli_max_history_drop_is_threaded_to_the_dry_run_rehearsal(tmp_path, capsys, monkeypatch):
    world = _world()
    cutoff = _iso(world["SPY"].index[-200])
    catalog_path, quotes_path, ledger_path, research_root, futures_dir = _fixture(tmp_path, world=world)
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(world))
    base = ["--dry-run", "--quotes", quotes_path, "--catalog", catalog_path, "--ledger", ledger_path,
            "--research-root", research_root, "--futures-dir", futures_dir, "--random-draws", "3",
            "--history-end", cutoff]
    assert driver.main(base + ["--max-history-drop", "5"]) == 0
    path = os.path.join(research_root, driver.DEFAULT_RUN_ID, "gold-copper-overlay-dry-run.json")
    with open(path, encoding="utf-8") as fh:
        assert "more than the 5" in json.load(fh)["result"]["history_end_guard_would_refuse"]
    assert driver.main(base + ["--max-history-drop", "199"]) == 0
    with open(path, encoding="utf-8") as fh:
        assert json.load(fh)["result"]["history_end_guard_would_refuse"] is None


@pytest.mark.parametrize("bad", ["-1", "abc", "2.5", ""])
def test_cli_max_history_drop_must_be_a_non_negative_whole_number(tmp_path, monkeypatch, bad):
    def refuse(*args, **kwargs):
        raise AssertionError("a bad --max-history-drop must be refused before any work is done")
    monkeypatch.setattr(driver, "compute_and_record", refuse)
    with pytest.raises(SystemExit) as excinfo:
        driver.main(["--dry-run", "--quotes", str(tmp_path / "q.json"), "--max-history-drop", bad]
                    + _cli_paths(tmp_path))
    assert excinfo.value.code == 2
    assert not (tmp_path / "cli_research").exists() and not (tmp_path / "cli_ledger.jsonl").exists()


@pytest.mark.parametrize("scenario", ["too_large", "after_tested_date"])
def test_cli_a_guard_refusal_exits_2_prints_the_standard_refusal_and_writes_nothing(
        tmp_path, capsys, monkeypatch, scenario):
    world = _world()
    sessions = world["SPY"].index
    cutoff = _iso(sessions[-200])
    if scenario == "too_large":
        extra, tested, code_expected = [], "2026-09-24", "history_end_drop_too_large"
    else:
        extra, tested = ["--max-history-drop", str(_HE_ACCEPT_ALL)], _iso(sessions[-201])
        code_expected = "history_end_after_tested_date"
    code, ledger_path, research_root = _real_cli(tmp_path, monkeypatch, world, "--history-end", cutoff, *extra,
                                                 tested_date=tested)
    assert code == 2
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["output"] is None and printed["result"]["reason_code"] == code_expected
    assert list(printed["result"]) == _HE_REFUSAL_KEYS
    assert not os.path.exists(ledger_path)
    for name in ("gold-copper-overlay-results.json", "gold-copper-overlay-dry-run.json"):
        assert not os.path.exists(os.path.join(research_root, "gcr-cli", name))
    assert os.path.isdir(_cache_dir(research_root, "gcr-cli"))          # the load step ran, so its cache is kept


def test_cli_a_not_covered_refusal_exits_2_and_keeps_the_cache(tmp_path, capsys, monkeypatch):
    code, ledger_path, research_root = _cli_history_end(tmp_path, monkeypatch, _world(), "2014-01-02")
    assert code == 2
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["result"]["reason_code"] == "history_end_not_covered"
    assert _HE_CACHE_SENTENCE in printed["result"]["reason"]
    assert os.path.isdir(_cache_dir(research_root, driver.DEFAULT_RUN_ID)) and not os.path.exists(ledger_path)


@pytest.mark.parametrize("extra", [[], ["--dry-run"]])
def test_cli_a_malformed_tested_date_under_a_history_end_exits_2_and_leaves_no_trace(
        tmp_path, capsys, monkeypatch, extra):
    world = _world()
    cutoff = _iso(world["SPY"].index[-200])
    code, ledger_path, research_root = _real_cli(tmp_path, monkeypatch, world, "--history-end", cutoff, *extra,
                                                 tested_date="2026-9-25")
    assert code == 2
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["output"] is None and printed["result"]["reason_code"] == "history_end_invalid"
    assert list(printed["result"]) == _HE_REFUSAL_KEYS and "tested_date" in printed["result"]["reason"]
    assert not os.path.exists(ledger_path)
    assert not os.path.exists(os.path.join(research_root, "gcr-cli"))     # no cache, no result file, no report


def test_cli_a_real_run_takes_the_explicit_drop_and_writes_the_setting_at_the_top_of_the_file(
        tmp_path, capsys, monkeypatch):
    world = _world()
    cutoff = _iso(_low_end_world()["SPY"].index[-1])
    dry, _, _ = _he_run(_mk(tmp_path, "dry"), cutoff, world=world)
    worst = _worst_drop(dry["history_end"])
    refused_code, refused_ledger, _ = _real_cli(_mk(tmp_path, "one_less"), monkeypatch, world, "--history-end", cutoff,
                                                "--max-history-drop", str(worst - 1))
    assert refused_code == 2 and not os.path.exists(refused_ledger)
    capsys.readouterr()
    code, ledger_path, research_root = _real_cli(_mk(tmp_path, "exact"), monkeypatch, world, "--history-end", cutoff,
                                                 "--max-history-drop", str(worst))
    assert code == 0
    with open(os.path.join(research_root, "gcr-cli", "gold-copper-overlay-results.json"), encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["history_end_requested"] == cutoff and payload["history_end"]["history_end"] == cutoff
    assert payload["result"]["stage"] in _SCORED and payload["result"]["history_end_requested"] == cutoff
    assert len([r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]) == 1


def test_cli_a_refusal_after_the_trim_writes_the_setting_at_the_top_of_the_file(tmp_path, monkeypatch):
    cut_world, _ = _open_window_world(3)
    cutoff = _iso(cut_world["SPY"].index[-1])
    code, ledger_path, research_root = _real_cli(tmp_path, monkeypatch, _world(), "--history-end", cutoff,
                                                 "--max-history-drop", str(_HE_ACCEPT_ALL))
    assert code == 0 and not os.path.exists(ledger_path)             # a non-history-end refusal exits 0
    with open(os.path.join(research_root, "gcr-cli", "gold-copper-overlay-results.json"), encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["result"]["reason_code"] == "truncated_final_window"
    assert payload["history_end_requested"] == cutoff and payload["result"]["history_end"]["history_end"] == cutoff


def test_cli_history_end_together_with_freeze_futures_is_refused_before_anything_is_fetched_or_written(
        tmp_path, monkeypatch):
    def refuse_network(symbol):
        raise AssertionError("--history-end with --freeze-futures must be refused before any fetch")
    monkeypatch.setattr(driver.mod, "real_yahoo_fetcher", refuse_network)
    with pytest.raises(SystemExit) as excinfo:
        driver.main(["--freeze-futures", "--through", "2013-12-20", "--history-end", "2013-12-20",
                     "--futures-dir", str(tmp_path / "f")])
    assert excinfo.value.code == 2 and not (tmp_path / "f").exists()


def test_cli_through_without_freeze_futures_is_not_a_usage_error_and_a_score_ignores_it(
        tmp_path, capsys, monkeypatch):
    # with the history end unset, --through alone is not a usage error: a score simply ignores it
    calls = []

    def stub(**kwargs):
        calls.append(kwargs)
        return {"family_id": _FAMILY, "reason_code": "not_in_catalog", "reason": "stubbed"}
    monkeypatch.setattr(driver, "compute_and_record", stub)
    code = driver.main(["--dry-run", "--quotes", str(tmp_path / "q.json"), "--through", "2013-12-20"]
                       + _cli_paths(tmp_path))
    assert code == 0 and len(calls) == 1 and calls[0]["history_end"] is None
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["result"]["reason_code"] == "not_in_catalog"


# ---- (c) doc truth

def test_the_driver_docs_tell_the_truth_about_what_a_refused_history_end_leaves_behind(capsys):
    text = " ".join(driver.__doc__.split())
    for code in hc.HISTORY_END_REASON_CODES:
        assert code in text, code
    assert "--max-history-drop" in text
    assert "no ledger row" in text and "no result file" in text and "no report" in text
    assert "per-run OHLC cache" in text and "fresh --run-id" in text
    assert "outcome row's account ends at the cutoff" in text and "NEW family's first_test pin" in text
    assert "record_first_test" in text and "no-op" in text
    with pytest.raises(SystemExit):
        driver.main(["--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "--max-history-drop" in help_text and "no ledger row" in help_text
    assert "cannot be re-scored on longer history" in help_text


def test_the_driver_docs_tell_the_truth_about_refusals_that_are_not_history_end_refusals_and_the_futures():
    text = " ".join(driver.__doc__.split())
    # main() writes its small results JSON and exits 0 for every refusal that is not a history_end_* one, and
    # the docstring says so
    start = text.index("``stale_ohlc_cache``")
    window = text[start:start + 260]
    assert "no ledger row" in window and "small results JSON" in window and "exits 0" in window
    assert "cache is kept" in window
    # the usage traps: --history-end with --freeze-futures is refused, --through alone is not
    assert "``--history-end`` cannot be combined with ``--freeze-futures`` (a usage error)" in text
    # the door checks the guard's own two inputs, and the futures wording matches what the note says
    assert "the same door validates the real-run guard's two inputs for real and dry runs alike" in text
    assert "unusable tested date" in text and "unusable ``max_history_drop``" in text
    assert f"{_FUTURES_LOADED_WHOLE} (so a malformed row dated after the cutoff could still refuse the run); " \
           f"{_FUTURES_ONE_ROW_PER_DECISION}" in text


def test_the_eight_digit_check_matches_standalone_numbers_but_not_hex_digests():
    assert _EIGHT_DIGITS.search("tested on " + "2026" + "0924" + " only")  # built from parts: this file holds no eight-digit run
    assert not _EIGHT_DIGITS.search("signal_module_sha256=a2e087c4" + "0c4d52497549d91ade")
    assert not _EIGHT_DIGITS.search("ratio 0.1234" + "5678 and 1.2026" + "0924")
