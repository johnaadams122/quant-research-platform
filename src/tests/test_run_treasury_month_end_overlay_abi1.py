"""Tests for scripts/run_treasury_month_end_overlay_abi1.py -- the bespoke
ABI-1 driver for treasury_month_end_index_demand_overlay.

Every run below uses SYNTHETIC price frames injected through the driver's
``ohlc_loader`` hook and a TEMPORARY ledger; nothing here reads the network or
touches data/mining/trial_ledger.jsonl. The dry-run tests prove wiring only --
never a real cost measurement (the quotes file is synthetic on purpose). The
family's catalog entry is read from data/mining/catalog.json.
"""
import json
import os
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

import run_treasury_month_end_overlay_abi1 as driver  # noqa: E402
from mining import account_relative, cost_tax_inputs, trial_ledger  # noqa: E402

_FAMILY = "treasury_month_end_index_demand_overlay"
_ENTRY_PATH = os.path.join(REPO_ROOT, "data", "mining", "catalog.json")  # the family's shipped catalog entry


def _catalog_entry_from(fh):
    """This family's entry in the shipped catalog."""
    return next(entry for entry in json.load(fh) if entry["family_id"] == _FAMILY)

# Deliberately DIFFERENT spreads per symbol everywhere below: with identical
# fixtures the suite would have zero power to catch a leg-cost / SPY-cost or a
# leg-frame / SPY-frame mixup.
_QUOTES = {
    "IEF": {"bid": 95.00, "ask": 95.02},     # ~2.1 bps
    "TLT": {"bid": 88.00, "ask": 88.05},     # ~5.7 bps
    "AGG": {"bid": 99.00, "ask": 99.03},     # ~3.0 bps
    "SPY": {"bid": 750.00, "ask": 750.15},   # ~2.0 bps
}


def _ohlc(close, index):
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.001,
                         "Low": np.minimum(open_, close) * 0.999, "Close": close}, index=index)


def _walk(index, seed, vol, drift, start):
    r = np.random.RandomState(seed).normal(drift, vol, len(index))
    r[0] = 0.0
    return start * np.cumprod(1.0 + r)


def _frames(n_legs=1900, lead=300, seed=5, irx_close=None, tail=0):
    """SPY has ``lead`` sessions of history before the legs list (production:
    SPY from 1993, AGG from 2003) and ends on the SAME last session as the legs
    (a run refuses unequal trailing coverage); only the financing series ^IRX
    runs ``tail`` sessions past the legs' last bar. Every frame sits on the
    REAL exchange session calendar (unscheduled closures such as the 2012
    Hurricane Sandy days carry no bar)."""
    sessions = pd.DatetimeIndex(mod_sessions(n_legs + lead + tail))
    account_end = len(sessions) - tail
    leg_idx = sessions[lead:account_end]
    spy_idx = sessions[:account_end]
    frames = {
        "SPY": _ohlc(_walk(spy_idx, seed, 0.01, 0.0004, 100.0), spy_idx),
        "IEF": _ohlc(_walk(leg_idx, seed + 1, 0.004, 0.0001, 100.0), leg_idx),
        "TLT": _ohlc(_walk(leg_idx, seed + 2, 0.009, 0.0001, 110.0), leg_idx),
        "AGG": _ohlc(_walk(leg_idx, seed + 3, 0.003, 0.0001, 90.0), leg_idx),
    }
    irx = np.linspace(0.5, 3.0, len(sessions)) if irx_close is None else irx_close(sessions)
    frames["^IRX"] = pd.DataFrame({"Open": irx, "High": irx, "Low": irx, "Close": irx}, index=sessions)
    return frames


def mod_sessions(n):
    from mining.batch5 import treasury_month_end_index_demand_overlay as mod
    return mod.exchange_sessions(pd.Timestamp("2008-01-02"), pd.Timestamp("2022-12-30"))[:n]


def _frames_cut_after_entry(past, *, window=-15):
    """The production-shaped frames with EVERY frame cut ``past`` sessions after
    one window's entry date: -1 = the session before the entry (window not
    started), 0 = data ends exactly on the entry date, 1 and 2 = inside the
    window (T-2, T-1), 3 = exactly on the exit (T)."""
    from mining.batch5 import treasury_month_end_index_demand_overlay as mod
    frames = _frames()
    leg_index = frames["IEF"].index
    entry, exit_ = mod.windows_in_data(leg_index)["windows"][window]
    last = leg_index[leg_index.get_loc(entry) + past]
    return {symbol: frame.loc[:last] for symbol, frame in frames.items()}, entry, exit_


def _loader(frames):
    def load(symbol):
        frame = frames[symbol].copy()
        frame.insert(0, "Date", frame.index)
        return frame.reset_index(drop=True)
    return load


def _quotes_payload(*, synthetic: bool) -> dict:
    return {
        "sampled_at_local": "2026-09-23T10:00:00",
        "market_state": "SYNTHETIC-test" if synthetic else "open",
        "source": "SYNTHETIC unit-test fixture" if synthetic else "unit test capture",
        "quotes": {s: {**q, "security_status": "Normal", "suspect": False} for s, q in _QUOTES.items()},
    }


def _fixture(tmp_path, *, synthetic=True, status="eligible", parameters=None):
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
        json.dump(_quotes_payload(synthetic=synthetic), fh)
    return catalog_path, quotes_path, str(tmp_path / "ledger.jsonl"), str(tmp_path / "research")


def _run(tmp_path, *, dry_run=True, synthetic=True, frames=None, run_id="tmo-test", **fixture_kwargs):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(
        tmp_path, synthetic=synthetic, **fixture_kwargs)
    result = driver.compute_and_record(
        catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path,
        run_id=run_id, tested_date="2026-09-23", dry_run=dry_run,
        signal_source_path="scripts/run_treasury_month_end_overlay_abi1.py",
        signal_source_sha256="deadbeef", research_root=research_root,
        ohlc_loader=_loader(frames or _frames()))
    return result, ledger_path


def _mk(tmp_path, name):
    path = tmp_path / name
    path.mkdir()
    return path


# ------------------------------------------------------------ dry-run wiring

def test_dry_run_evaluates_end_to_end_without_writing_a_ledger_row(tmp_path):
    result, ledger_path = _run(tmp_path)
    assert result["family_id"] == _FAMILY
    assert "reason_code" not in result        # no skip, no compute_error
    assert result["verdict"] in ("not_scoreable", "cut_dsr", "cut_risk_refusal",
                                 "holdout_not_evaluable", "cut_holdout", "survived")
    assert result["tax_policy"] == "realization"
    assert result["benchmark"] == "SPY"
    assert result["tickers"] == ["IEF", "TLT", "AGG"]
    assert not os.path.exists(ledger_path)    # dry run never writes


def test_dry_run_reports_the_structural_details(tmp_path):
    result, _ = _run(tmp_path)
    diag = result["overlay_diagnostics"]
    assert diag["n_windows"] == result["n_trades"] > 50
    assert diag["worst_window_net_ret"] < diag["best_window_net_ret"]
    assert diag["overlay_notional_fraction"] == 0.5 and diag["window_sessions"] == 3
    assert diag["first_window_entry"] > "2008-12"          # inside the legs' span, after the lead
    assert diag["windows_skipped_before_data"] in (0, 1)
    # the fixture sits on the real exchange sessions, so the October 2012 window
    # (entry the 26th, exit the 31st) is held across the Sandy closure: its 29th
    # and 30th are scheduled but not data sessions, so 2 fewer sessions are financed
    sandy = 2
    if diag["final_window_truncated"]:
        assert 3 * (diag["n_windows"] - 1) - sandy < diag["overlay_sessions"] < 3 * diag["n_windows"] - sandy
    else:
        assert diag["overlay_sessions"] == 3 * diag["n_windows"] - sandy
    assert 0.10 < diag["overlay_session_share"] < 0.17     # about 3 of every 21 sessions
    assert result["time_in_market"] == 1.0                 # the SPY core is always on
    json.dumps(result, allow_nan=False, default=str)       # nothing non-finite leaks


def test_the_account_lives_on_the_legs_span_not_on_spys_longer_history(tmp_path):
    frames = _frames()
    result, _ = _run(tmp_path, frames=frames)
    assert result["account_start"] == frames["IEF"].index[0].strftime("%Y-%m-%d")
    assert result["account_end"] == frames["IEF"].index[-1].strftime("%Y-%m-%d")
    assert frames["SPY"].index[0] < frames["IEF"].index[0]


def test_the_account_equals_the_evaluators_benchmark_series_outside_the_windows_in_the_production_shape():
    """SPY lists before the legs (as in production). Outside the windows the
    scored account must equal, bit for bit, the benchmark series the ABI-1
    evaluator subtracts (computed on SPY's FULL history); a core built from SPY
    sliced to the account span would differ on the first account day."""
    built = _built()
    frames, curve, bench = built[0], built[1], built[5]
    account = curve["strategy_daily"]
    outside = ~curve["overlay_mask"].to_numpy()
    assert frames["SPY"].index[0] < account.index[0]          # SPY really is longer
    assert outside[0] and outside.sum() > 1000
    np.testing.assert_array_equal(account.to_numpy()[outside],
                                  bench.reindex(account.index).to_numpy()[outside])


def test_costs_come_from_the_right_symbols_quotes(tmp_path):
    result, _ = _run(tmp_path)
    diag = result["overlay_diagnostics"]
    for symbol in ("IEF", "TLT", "AGG"):
        expected = cost_tax_inputs.compute_spread_bps(_QUOTES[symbol]["bid"], _QUOTES[symbol]["ask"])
        assert diag["leg_cost_bps"][symbol] == pytest.approx(expected)
    spy_bps = cost_tax_inputs.compute_spread_bps(_QUOTES["SPY"]["bid"], _QUOTES["SPY"]["ask"])
    assert diag["benchmark_entry_cost_bps"] == pytest.approx(spy_bps / 2.0)
    assert len({round(v, 6) for v in diag["leg_cost_bps"].values()}) == 3     # three distinct fixtures


def test_the_financing_rate_is_the_bill_series_not_a_constant(tmp_path):
    frames = _frames(irx_close=lambda idx: np.full(len(idx), 4.0))
    result, _ = _run(tmp_path, frames=frames)
    expected = float(account_relative.coupon_equivalent_yield(0.04))
    assert result["overlay_diagnostics"]["mean_financing_rate_annual_on_overlay_sessions"] == pytest.approx(
        expected, rel=1e-12)


def test_a_negative_bill_print_is_loaded_without_the_positivity_gate(tmp_path):
    def with_negative_prints(idx):
        irx = np.full(len(idx), 1.0)
        irx[(idx >= "2012-03-19") & (idx <= "2012-03-27")] = -0.105    # inside the data, at a window entry
        return irx
    frames = _frames(irx_close=with_negative_prints)
    assert (frames["^IRX"]["Close"] < 0).any()               # the fixture really carries negative prints
    result, _ = _run(tmp_path, frames=frames)
    assert "reason_code" not in result and np.isfinite(result["sharpe_daily_excess"])


def test_a_traded_leg_with_a_non_positive_price_still_fails_closed(tmp_path):
    frames = _frames()
    frames["AGG"].iloc[500, frames["AGG"].columns.get_loc("Close")] = -1.0
    with pytest.raises(ValueError, match="AGG"):
        _run(tmp_path, frames=frames)


def test_the_leg_frames_really_drive_the_curve(tmp_path):
    a, _ = _run(_mk(tmp_path, "a"), frames=_frames(seed=5))
    other = _frames(seed=5)
    other["TLT"] = _ohlc(_walk(other["TLT"].index, 77, 0.02, 0.0005, 110.0), other["TLT"].index)
    b, _ = _run(_mk(tmp_path, "b"), frames=other)
    assert a["sharpe_daily_excess"] != b["sharpe_daily_excess"]


def test_the_spy_frame_really_drives_the_core_and_the_benchmark(tmp_path):
    a, _ = _run(_mk(tmp_path, "a"), frames=_frames(seed=5))
    other = _frames(seed=5)
    other["SPY"] = _ohlc(_walk(other["SPY"].index, 41, 0.012, 0.0002, 100.0), other["SPY"].index)
    b, _ = _run(_mk(tmp_path, "b"), frames=other)
    assert a["sharpe_daily_excess"] != b["sharpe_daily_excess"]
    assert a["overlay_diagnostics"]["scored_excess_beta_to_spy"] != b["overlay_diagnostics"]["scored_excess_beta_to_spy"]


def test_the_scored_result_is_the_taxed_realization_evaluation_not_the_untaxed_one(tmp_path):
    """The recorded verdict must come from the realization-tax evaluation. A
    driver that scored the untaxed curve would still report tax_policy
    'realization' and still pass every wiring check above."""
    built = _built()
    taxed, untaxed = built[2], built[3]
    result, _ = _run(tmp_path)
    assert taxed["layer1"]["sharpe_daily_excess"] != pytest.approx(untaxed["layer1"]["sharpe_daily_excess"], rel=1e-9)
    assert result["sharpe_daily_excess"] == pytest.approx(taxed["layer1"]["sharpe_daily_excess"], rel=1e-12)
    assert result["dsr_value"] == pytest.approx(taxed["layer1"]["dsr_value"], rel=1e-9)
    assert result["verdict"] == taxed["verdict"]


def test_dry_run_reports_the_risk_layer_numbers_including_the_absolute_cap(tmp_path):
    result, _ = _run(tmp_path)
    risk = result["overlay_diagnostics"]["risk_layer"]
    assert risk["account_basis"] == "overlay_only_excess"     # overlay-only excess account basis
    assert risk["absolute_cap"] == -0.35
    assert risk["account_max_drawdown"] <= 0 and risk["benchmark_max_drawdown"] <= 0
    assert risk["absolute_passed"] == (risk["account_max_drawdown"] >= risk["absolute_cap"])
    assert risk["benchmark_itself_breaches_the_absolute_cap"] == (
        risk["benchmark_max_drawdown"] < risk["absolute_cap"])


def test_dry_run_reports_all_four_risk_refusals_not_only_the_absolute_cap(tmp_path):
    result, _ = _run(tmp_path)
    risk = result["overlay_diagnostics"]["risk_layer"]
    for key in ("relative_passed", "red_week_streak_passed", "worst_stretch_passed",
                "all_refusals_passed"):
        assert isinstance(risk[key], bool), key
    assert isinstance(risk["failed_refusals"], list)
    assert (risk["failed_refusals"] == []) == risk["all_refusals_passed"]
    assert risk["worst_stretch_window_weeks"] == 8


def test_the_excess_is_exactly_zero_outside_the_windows_through_the_real_tax_path_and_entry_cost():
    """The overlay-only excess account is only the overlay's dollars if the core
    IS the benchmark. Pinned end to end in the production shape (non-zero SPY
    entry cost, realization tax applied to every session's growth factor,
    SPY listing before the legs): a float-noise excess here would read as
    spurious losing weeks in the streak refusal."""
    built = _built()
    curve, taxed, bench = built[1], built[2], built[5]
    account = taxed["account_daily"]
    excess = account - bench.reindex(account.index)
    outside = ~curve["overlay_mask"].to_numpy()
    assert outside.sum() > 1000 and built[7] > 0.0            # a real entry cost was charged
    assert (excess[outside] == 0.0).all()
    assert (excess[curve["overlay_mask"].to_numpy()] != 0.0).any()


def test_a_core_that_differs_from_the_benchmark_outside_the_windows_is_refused(tmp_path, monkeypatch):
    """The excess is only overlay-only if the account is the benchmark whenever
    the overlay is off. A curve whose core drifted from the benchmark (here by
    one 0.01 percent day) would be scored on a contaminated excess: refused
    before anything is recorded."""
    real = driver.mod.build_curve

    def drifted(*args, **kwargs):
        curve = dict(real(*args, **kwargs))
        daily = curve["strategy_daily"].copy()
        off = np.flatnonzero(~curve["overlay_mask"].to_numpy())
        daily.iloc[off[700]] += 1e-4
        curve["strategy_daily"] = daily
        return curve
    monkeypatch.setattr(driver.mod, "build_curve", drifted)
    with pytest.raises(RuntimeError, match="not overlay-only"):
        _run(tmp_path, dry_run=False, synthetic=False)
    assert not os.path.exists(str(tmp_path / "ledger.jsonl"))


def test_the_gated_risk_layer_is_the_excess_account_and_the_whole_account_is_a_separate_reference():
    """Overlay-only excess account basis. Distinct numbers everywhere: the gated block must
    read the overlay-only EXCESS account, the reference block the WHOLE account,
    and neither may be a relabelled copy of the other."""
    from mining import benchmark_relative
    built = _built()
    taxed, bench = built[2], built[5]
    account = taxed["account_daily"]
    excess = account - bench.reindex(account.index)
    risk = _diagnostics(built)["risk_layer"]
    ref = risk["whole_account_reference"]
    assert risk["account_max_drawdown"] == pytest.approx(benchmark_relative.max_drawdown(excess), abs=1e-12)
    assert ref["account_max_drawdown"] == pytest.approx(benchmark_relative.max_drawdown(account), abs=1e-12)
    assert risk["account_max_drawdown"] != pytest.approx(ref["account_max_drawdown"])
    assert ref["basis"] == "whole_account" and ref["gates_verdict"] is False
    direct = account_relative.risk_refusals(account, bench)
    assert ref["absolute_passed"] == direct["drawdown"]["absolute_passed"]
    assert ref["all_refusals_passed"] == direct["all_refusals_passed"]
    # the SPY-core ceiling (not applied to the gated block) is real on the whole account, and
    # it is the whole-account reference that shows it, not the gated block
    assert ref["benchmark_max_drawdown"] == pytest.approx(risk["benchmark_max_drawdown"])


def test_a_curve_that_lost_its_account_kind_is_refused_before_anything_is_recorded(tmp_path, monkeypatch):
    """The ledger note states the risk basis. If the curve stopped declaring
    itself a benchmark-plus-overlay account, evaluate_family would silently
    score the whole account while the note claimed the excess account."""
    real = driver.mod.build_curve

    def stripped(*args, **kwargs):
        curve = dict(real(*args, **kwargs))
        curve.pop("account_kind")
        return curve
    monkeypatch.setattr(driver.mod, "build_curve", stripped)
    with pytest.raises(RuntimeError, match="overlay_only_excess"):
        _run(tmp_path, dry_run=False, synthetic=False)
    assert not os.path.exists(str(tmp_path / "ledger.jsonl"))


def test_the_risk_layer_report_reads_the_spy_relative_refusals_from_layer_two():
    """Distinct values everywhere, so a swapped key or a wrong source cannot pass."""
    import copy
    built = _built()
    taxed = copy.deepcopy(built[2])
    taxed["layer2"]["red_week_streak"].update(passed=False, account=7, benchmark=5)
    taxed["layer2"]["worst_stretch"].update(passed=True, account=-0.0421, benchmark=-0.0313, window_weeks=8)
    risk = _diagnostics(built, result=taxed)["risk_layer"]
    assert risk["red_week_streak_passed"] is False
    assert (risk["red_week_streak_account"], risk["red_week_streak_benchmark"]) == (7, 5)
    assert risk["worst_stretch_passed"] is True
    assert risk["worst_stretch_account"] == pytest.approx(-0.0421)
    assert risk["worst_stretch_benchmark"] == pytest.approx(-0.0313)


@pytest.mark.parametrize("which", ["relative_drawdown", "absolute_drawdown_cap", "red_week_streak",
                                   "worst_8_week_stretch"])
def test_the_risk_layer_report_names_exactly_the_refusal_that_failed(which):
    import copy
    built = _built()
    taxed = copy.deepcopy(built[2])
    layer2 = taxed["layer2"]
    # start from a result in which nothing fails ...
    layer2["drawdown"].update(relative_passed=True, absolute_passed=True, passed=True)
    layer2["red_week_streak"]["passed"] = True
    layer2["worst_stretch"]["passed"] = True
    layer2["all_refusals_passed"] = True
    clean = _diagnostics(built, result=taxed)["risk_layer"]
    assert clean["failed_refusals"] == [] and clean["all_refusals_passed"] is True
    # ... then break exactly one
    if which == "relative_drawdown":
        layer2["drawdown"].update(relative_passed=False, passed=False)
    elif which == "absolute_drawdown_cap":
        layer2["drawdown"].update(absolute_passed=False, passed=False)
    elif which == "red_week_streak":
        layer2["red_week_streak"]["passed"] = False
    else:
        layer2["worst_stretch"]["passed"] = False
    layer2["all_refusals_passed"] = False
    risk = _diagnostics(built, result=taxed)["risk_layer"]
    assert risk["failed_refusals"] == [which]
    assert risk["all_refusals_passed"] is False


def test_dry_run_reports_the_disclosed_next_open_sensitivity(tmp_path):
    result, _ = _run(tmp_path)
    sens = result["overlay_diagnostics"]["next_open_sensitivity"]
    assert "next-open" in sens["convention"]
    assert sens["sharpe_daily_excess"] is not None
    assert sens["sharpe_daily_excess"] != result["sharpe_daily_excess"]     # a genuinely different curve
    assert sens["verdict"] in ("not_scoreable", "cut_dsr", "cut_risk_refusal",
                               "holdout_not_evaluable", "cut_holdout", "survived")


def test_dry_run_reports_the_tax_identity_approximation_it_relies_on(tmp_path):
    result, _ = _run(tmp_path)
    diag = result["overlay_diagnostics"]
    assert 0.5 < diag["tax_identity_ratio_min"] <= diag["tax_identity_ratio_max"] < 2.0
    taxed, untaxed = diag["sharpe_daily_excess_taxed"], diag["sharpe_daily_excess_untaxed"]
    assert taxed == pytest.approx(result["sharpe_daily_excess"])
    effect = diag["tax_effect_on_daily_excess_sharpe"]
    assert effect == pytest.approx(taxed - untaxed)
    worst = max(abs(diag["tax_identity_ratio_min"] - 1.0), abs(diag["tax_identity_ratio_max"] - 1.0))
    assert diag["tax_identity_sensitivity_on_sharpe"] == pytest.approx(worst * abs(effect))
    assert "tax_identity_effect_bound_on_sharpe" not in diag       # no overclaiming key


# ---------------------------------------------------------------- guards

def test_real_run_refuses_a_synthetic_quotes_file(tmp_path):
    with pytest.raises(ValueError, match="SYNTHETIC"):
        _run(tmp_path, dry_run=False, synthetic=True)


def test_a_missing_quote_for_any_leg_is_refused(tmp_path):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path)
    payload = _quotes_payload(synthetic=True)
    del payload["quotes"]["AGG"]
    with open(quotes_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    with pytest.raises(ValueError, match="AGG"):
        driver.compute_and_record(
            catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path,
            run_id="x", tested_date="2026-09-23", dry_run=True, signal_source_path="p",
            signal_source_sha256="s", research_root=research_root, ohlc_loader=_loader(_frames()))


def test_missing_catalog_entry_and_ineligible_status_skip_cleanly(tmp_path):
    result, _ = _run(_mk(tmp_path, "inelig"), status="ruled_out")
    assert result["reason_code"] == "catalog_status"
    catalog_path, quotes_path, ledger_path, research_root = _fixture(_mk(tmp_path, "none"))
    with open(catalog_path, "w", encoding="utf-8") as fh:
        json.dump([], fh)
    skipped = driver.compute_and_record(
        catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path,
        run_id="x", tested_date="2026-09-23", dry_run=True,
        signal_source_path="p", signal_source_sha256="s", research_root=research_root,
        ohlc_loader=_loader(_frames()))
    assert skipped["reason_code"] == "not_in_catalog"


def test_parameter_drift_blocks_the_run(tmp_path):
    with open(_ENTRY_PATH, "r", encoding="utf-8") as fh:
        drifted = dict(_catalog_entry_from(fh)["parameters"])
    drifted["window_sessions"] = 4
    result, _ = _run(tmp_path, parameters=drifted)
    assert result["reason_code"] == "parameter_drift"


def test_an_existing_abi1_outcome_blocks_a_second_trial(tmp_path):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path)
    trial_ledger.record_outcome(ledger_path, _FAMILY, "earlier", "2026-09-01",
                                stage="cut_dsr", dsr_value=0.1, sharpe_in_sample=0.0,
                                construction="ABI-1", note="prior")
    result = driver.compute_and_record(
        catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path,
        run_id="again", tested_date="2026-09-23", dry_run=True,
        signal_source_path="p", signal_source_sha256="s", research_root=research_root,
        ohlc_loader=_loader(_frames()))
    assert result["reason_code"] == "already_scored_abi1"


# ---------------------------------------------------- ledger path (tmp ledger)

def test_a_non_dry_run_against_a_temp_ledger_writes_exactly_one_abi1_row_with_disclosures(tmp_path):
    result, ledger_path = _run(tmp_path, dry_run=False, synthetic=False)
    rows = trial_ledger.read_all(ledger_path)
    outcomes = [r for r in rows if r.get("event") == "outcome"]
    assert len(outcomes) == 1
    assert sorted(r.get("event") for r in rows) == ["first_test", "outcome"]
    row = outcomes[0]
    assert row["family_id"] == _FAMILY and row["construction"] == "ABI-1"
    # the fixture scores (fixed seed), so the disclosure note IS written --
    # asserted, not skipped, so this check can never pass vacuously
    assert result["stage"] in ("survived", "cut_dsr", "cut_holdout", "cut_risk_refusal")
    assert row["stage"] == result["stage"]
    assert row["dsr_value"] == pytest.approx(result["dsr_value"], rel=1e-12)
    assert row["sharpe_in_sample"] == pytest.approx(result["sharpe_daily_excess"], rel=1e-12)
    note = (row.get("note") or "").lower()
    # the template prefix ABI-1 writes on every scored row: WHAT the score was
    # computed against and under which tax policy -- a permanent, unrepairable label
    assert note.startswith("cash_convention=sweep_cash_daily; benchmark=spy; tax_policy=realization; cost model:")
    for phrase in ("not modelled", "margin rate", "bill rate plus a spread",
                   "no spread is charged", "flatters the strategy",
                   "risk layer (the risk-basis rule", "overlay-only excess account",
                   "risk_account_basis=overlay_only_excess", "whole-account reference",
                   "gating nothing", "0.35 absolute drawdown cap",
                   "disclosed sensitivity", "next-open",
                   "regular-holiday schedule", "never read off the data",
                   "heuristic sensitivity", "different strategy definition",
                   "tax walk", "price data: ief fetched", "spy fetched", "^irx fetched",
                   # pre-registration honesty: not every constant was chosen blind
                   "pre-registration limits", "not all chosen blind",
                   "fixed after descriptive looks", "headline example", "outside the trial ledger",
                   # the three SPY-relative refusals, not only the absolute cap
                   "spy-relative refusals", "longest losing-week streak", "worst 8-week stretch",
                   "much less binding", "refusals failed in this run",
                   # the scale of the excess account and the tax term are stated
                   "in units of whole-account equity", "realization-tax drag",
                   # the ordering clause: the rule is not certified blind to this family, whose
                   # dry-run path can also report whole-account risk numbers
                   "dry-run path",
                   # window coverage reaches the permanent note, not only the JSON report
                   "windows_skipped_before_data=", "windows_not_started=",
                   "final_window_truncated=false", "window_entered_on_last_bar=false",
                   "account span 2009-03-12..", "latest-listing leg's first session"):
        assert phrase in note, phrase
    # this run's NUMBERS and the whole-account failed list are in the
    # permanent note, not just its wording (a driver that dropped them would otherwise go unnoticed)
    risk = result["overlay_diagnostics"]["risk_layer"]
    ref = risk["whole_account_reference"]
    assert f"max drawdown {round(risk['account_max_drawdown'], 4)} against spy's {round(risk['benchmark_max_drawdown'], 4)}" in note
    assert f"streak {risk['red_week_streak_account']} against {risk['red_week_streak_benchmark']}" in note
    assert f"max drawdown {round(ref['account_max_drawdown'], 4)} against spy's" in note
    assert "whole-account refusals failed: " + (", ".join(ref["failed_refusals"]) or "none") in note
    assert "refusals failed in this run: " + (", ".join(risk["failed_refusals"]) or "none") in note
    assert os.path.abspath(ledger_path).startswith(os.path.abspath(str(tmp_path)))    # never the real ledger


def test_an_unscoreable_real_run_records_the_instrument_set_benchmark_and_cash_convention(tmp_path):
    """The two UNSCORED verdicts carry what was scored in the ledger row's
    evidence rather than in its note (a short history here is not scoreable)."""
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path, synthetic=False)
    result = driver.compute_and_record(
        catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path,
        run_id="short-history", tested_date="2026-09-23", dry_run=False,
        signal_source_path="scripts/run_treasury_month_end_overlay_abi1.py",
        signal_source_sha256="a" * 64, research_root=research_root,
        ohlc_loader=_loader(_frames(n_legs=220)))
    assert result["verdict"] == "not_scoreable" and result["stage"] == "insufficient_evidence"
    outcomes = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    assert len(outcomes) == 1
    observed = outcomes[0]["evidence"]["observed"]
    assert observed["instrument_set"] == ["IEF", "TLT", "AGG", "SPY"]      # legs AND the benchmark core
    assert observed["benchmark"] == "SPY" and observed["cash_convention"] == "sweep_cash_daily"
    with open(_ENTRY_PATH, "r", encoding="utf-8") as fh:
        assert observed["parameters"] == _catalog_entry_from(fh)["parameters"]
    assert outcomes[0]["construction"] == "ABI-1" and outcomes[0]["family_id"] == _FAMILY


def test_the_driver_default_ledger_is_never_used_by_these_tests():
    real = os.path.abspath(driver.LEDGER_PATH)
    assert real.endswith(os.path.join("data", "mining", "trial_ledger.jsonl"))


# ------------------------------------------------- window coverage

@pytest.mark.parametrize("past,truncated,on_last_bar,not_started", [
    (-1, False, False, 1),       # data ends the session before an entry: that window has not started
    (0, False, True, 0),         # data ends exactly on an entry date: it enters and opens nothing
    (1, True, False, 0),         # T-2: a real trade is closed early at the last close
    (2, True, False, 0),         # T-1
    (3, False, False, 0),        # T: the window completes exactly
])
def test_the_window_coverage_flags_reach_the_report_and_the_note(tmp_path, past, truncated, on_last_bar,
                                                                  not_started):
    frames, entry, exit_ = _frames_cut_after_entry(past)
    result, _ = _run(tmp_path, frames=frames)
    diag = result["overlay_diagnostics"]
    assert result["account_end"] == frames["IEF"].index[-1].strftime("%Y-%m-%d")
    assert diag["final_window_truncated"] is truncated
    assert diag["window_entered_on_last_bar"] is on_last_bar
    assert diag["windows_not_started"] == not_started
    note = result["cost_model_note"]
    assert f"final_window_truncated={truncated}" in note
    assert f"window_entered_on_last_bar={on_last_bar}" in note
    assert f"windows_not_started={not_started}" in note
    assert f"windows_skipped_before_data={diag['windows_skipped_before_data']}" in note
    # the trades frame never carries a window that opened nothing
    last_exit = pd.Timestamp(diag["last_window_exit"])
    assert (last_exit == frames["IEF"].index[-1]) if truncated else (last_exit <= frames["IEF"].index[-1])


@pytest.mark.parametrize("past", [1, 2])
def test_a_real_run_refuses_a_truncated_final_window_and_writes_nothing(tmp_path, past):
    """A window still open at the last bar would be closed early at the last
    close (a partial bar could be baked into that exit price) and would not be
    the pre-registered T exit; the permanent one-shot row must never carry it."""
    frames, _, _ = _frames_cut_after_entry(past)
    result, ledger_path = _run(tmp_path, dry_run=False, synthetic=False, frames=frames)
    assert result["reason_code"] == "truncated_final_window"
    assert "verdict" not in result
    assert not os.path.exists(ledger_path)


@pytest.mark.parametrize("past,on_last_bar", [(0, True), (3, False)])
def test_a_real_run_on_complete_data_records_the_window_coverage_in_the_ledger_note(tmp_path, past, on_last_bar):
    frames, _, _ = _frames_cut_after_entry(past)
    result, ledger_path = _run(tmp_path, dry_run=False, synthetic=False, frames=frames)
    assert result["stage"] in ("survived", "cut_dsr", "cut_holdout", "cut_risk_refusal")
    outcomes = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    assert len(outcomes) == 1
    note = outcomes[0]["note"].lower()
    assert "final_window_truncated=false" in note
    assert f"window_entered_on_last_bar={str(on_last_bar).lower()}" in note
    assert "windows_not_started=0" in note and "windows_skipped_before_data=" in note


# --------------------------------------------- data-calendar gaps are refused, not trimmed

@pytest.mark.parametrize("dry_run", [True, False])
def test_unequal_trailing_coverage_is_refused_with_a_reason_code_and_the_missing_date(tmp_path, dry_run):
    """IEF and AGG lack the last bar that TLT and SPY have (the vendor shape seen
    on a vendor data hole). The account must not silently end a session early."""
    frames = _frames()
    last = frames["TLT"].index[-1]
    frames["IEF"] = frames["IEF"].iloc[:-1]
    frames["AGG"] = frames["AGG"].iloc[:-1]
    result, ledger_path = _run(tmp_path, dry_run=dry_run, synthetic=dry_run, frames=frames)
    assert result["reason_code"] == "unequal_trailing_coverage"
    assert result["missing_dates"] == [last.strftime("%Y-%m-%d")]
    assert result["n_missing_dates"] == 1
    assert last.strftime("%Y-%m-%d") in result["reason"] and "IEF" in result["reason"]
    assert "verdict" not in result and not os.path.exists(ledger_path)
    json.dumps(result, allow_nan=False)                                # a clean JSON refusal


@pytest.mark.parametrize("dry_run", [True, False])
def test_an_ordinary_session_missing_from_every_series_is_refused_and_named(tmp_path, dry_run):
    """Every pairwise calendar check passes when ALL series lack the same session,
    and its multi-day close-to-close move would be treated as one bar."""
    frames = _frames()
    gone = frames["IEF"].index[900]
    frames = {symbol: frame.drop(gone) for symbol, frame in frames.items()}
    result, ledger_path = _run(tmp_path, dry_run=dry_run, synthetic=dry_run, frames=frames)
    assert result["reason_code"] == "missing_exchange_session"
    assert result["missing_dates"] == [gone.strftime("%Y-%m-%d")]
    assert gone.strftime("%Y-%m-%d") in result["reason"]
    assert "verdict" not in result and not os.path.exists(ledger_path)


def test_a_known_unscheduled_closure_is_no_gap_and_a_bar_on_it_is_refused(tmp_path):
    """The 2012 Hurricane Sandy days: absent from the data is the truth (the
    fixture frames are built that way and score), present is a vendor defect."""
    frames = _frames()
    sandy = pd.Timestamp("2012-10-29")
    assert sandy not in frames["IEF"].index
    ok, _ = _run(_mk(tmp_path, "ok"), frames=frames)
    assert "reason_code" not in ok
    stray = {}
    for symbol, frame in frames.items():
        row = frame.loc[[frame.index[frame.index > sandy][0]]].copy()
        row.index = pd.DatetimeIndex([sandy])
        stray[symbol] = pd.concat([frame, row]).sort_index()
    refused, _ = _run(_mk(tmp_path, "stray"), frames=stray)
    assert refused["reason_code"] == "bar_on_closed_day"
    assert refused["extra_dates"] == ["2012-10-29"] and "2012-10-29" in refused["reason"]


# ----------------------------------------- a real run never reuses cache

def test_a_real_run_refuses_ohlc_cached_by_an_earlier_dry_run_under_the_same_run_id(tmp_path):
    dry, _ = _run(tmp_path)                                            # caches OHLC under research-root/run-id
    assert set(dry["ohlc_sources"].values()) == {"fetched"}
    real, ledger_path = _run(tmp_path, dry_run=False, synthetic=False)  # SAME run id: a silent cache hit
    assert real["reason_code"] == "stale_ohlc_cache"
    assert "fresh --run-id" in real["reason"]
    assert real["cached_symbols"] == ["IEF", "TLT", "AGG", "SPY", "^IRX"]
    assert "verdict" not in real
    assert not os.path.exists(ledger_path)                              # nothing permanent was written


def test_a_real_run_names_only_the_symbols_that_were_cached(tmp_path):
    _run(tmp_path)
    cache_dir = os.path.join(str(tmp_path / "research"), "tmo-test", "artifacts", "ohlc")
    for name in os.listdir(cache_dir):
        if name != "TLT.csv":
            os.remove(os.path.join(cache_dir, name))
    real, ledger_path = _run(tmp_path, dry_run=False, synthetic=False)
    assert real["reason_code"] == "stale_ohlc_cache" and real["cached_symbols"] == ["TLT"]
    assert not os.path.exists(ledger_path)


def test_a_real_run_with_a_fresh_run_id_after_a_dry_run_is_scored_from_freshly_fetched_history(tmp_path):
    _run(tmp_path)
    real, ledger_path = _run(tmp_path, dry_run=False, synthetic=False, run_id="tmo-fresh")
    assert "reason_code" not in real
    assert set(real["ohlc_sources"].values()) == {"fetched"}
    assert len([r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]) == 1


# ------------------------------------------------------------------- cli

def test_cli_dry_run_writes_a_json_result_and_exits_zero(tmp_path, capsys, monkeypatch):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path)
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(_frames()))
    code = driver.main(["--quotes", quotes_path, "--dry-run", "--catalog", catalog_path,
                        "--ledger", ledger_path, "--research-root", research_root,
                        "--run-id", "cli-test"])
    assert code == 0
    output = os.path.join(research_root, "cli-test", "treasury-month-end-overlay-dry-run.json")
    with open(output, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["dry_run"] is True and payload["construction"] == "ABI-1"
    assert payload["result"]["family_id"] == _FAMILY
    assert not os.path.exists(ledger_path)


def test_a_real_run_requires_an_explicit_tested_date(tmp_path, monkeypatch):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path, synthetic=False)
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(_frames()))
    with pytest.raises(SystemExit):
        driver.main(["--quotes", quotes_path, "--catalog", catalog_path,
                     "--ledger", ledger_path, "--research-root", research_root,
                     "--run-id", "no-date"])
    assert not os.path.exists(ledger_path)
    assert not os.path.exists(os.path.join(research_root, "no-date"))


def test_a_real_run_requires_an_explicit_run_id(tmp_path, monkeypatch):
    """The shared default run id is where a dry run caches its OHLC; a real
    one-shot run must name its own."""
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path, synthetic=False)
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(_frames()))
    with pytest.raises(SystemExit):
        driver.main(["--quotes", quotes_path, "--catalog", catalog_path, "--ledger", ledger_path,
                     "--research-root", research_root, "--tested-date", "2026-09-23"])
    assert not os.path.exists(ledger_path)
    assert not os.path.exists(research_root)


def test_a_dry_run_without_a_run_id_falls_back_to_the_default_run_id(tmp_path, monkeypatch):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path)
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(_frames()))
    assert driver.main(["--quotes", quotes_path, "--dry-run", "--catalog", catalog_path,
                        "--ledger", ledger_path, "--research-root", research_root]) == 0
    assert os.path.exists(os.path.join(research_root, driver.DEFAULT_RUN_ID,
                                       "treasury-month-end-overlay-dry-run.json"))
    assert not os.path.exists(ledger_path)


def test_result_reports_the_provenance_of_every_price_history(tmp_path):
    """A DRY run under a re-used run_id re-uses OHLC cached by an earlier run (a
    real run refuses that, see the stale-cache tests); the result must say where
    each history came from and where each ends, so a stale cache is visible."""
    frames = _frames(tail=5)             # ^IRX runs 5 sessions past the legs' (and SPY's) last bar
    first, _ = _run(_mk(tmp_path, "one"), frames=frames)
    assert set(first["ohlc_sources"]) == {"IEF", "TLT", "AGG", "SPY", "^IRX"}
    assert set(first["ohlc_sources"].values()) == {"fetched"}
    for symbol in ("IEF", "TLT", "AGG", "SPY", "^IRX"):
        assert first["last_bar_dates"][symbol] == frames[symbol].index[-1].strftime("%Y-%m-%d"), symbol
    assert first["last_bar_dates"]["^IRX"] != first["last_bar_dates"]["AGG"]
    catalog_path, quotes_path, ledger_path, research_root = _fixture(_mk(tmp_path, "two"))
    common = dict(catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path,
                  run_id="tmo-test", tested_date="2026-09-23", dry_run=True,
                  signal_source_path="p", signal_source_sha256="s", research_root=research_root)
    driver.compute_and_record(**common, ohlc_loader=_loader(frames))
    again = driver.compute_and_record(**common, ohlc_loader=_loader(frames))
    assert set(again["ohlc_sources"].values()) == {"cache"}
    os.remove(os.path.join(research_root, "tmo-test", "artifacts", "ohlc", "TLT.csv"))
    mixed = driver.compute_and_record(**common, ohlc_loader=_loader(frames))
    assert mixed["ohlc_sources"]["TLT"] == "fetched" and mixed["ohlc_sources"]["IEF"] == "cache"


# --------------------------------------------- reported details vs independent maths

def _built():
    from mining import abi1_driver, benchmark_relative
    from mining.batch5 import treasury_month_end_index_demand_overlay as mod
    frames = _frames()
    costs = {s: cost_tax_inputs.compute_spread_bps(q["bid"], q["ask"]) for s, q in _QUOTES.items()}
    leg_costs = {s: costs[s] for s in mod.TRADED}
    entry_cost = costs["SPY"] / 2.0
    legs = {s: frames[s] for s in mod.TRADED}
    curve = mod.build_curve(frames["SPY"], legs, frames["^IRX"], leg_cost_bps=leg_costs,
                            spy_entry_cost_bps=entry_cost)
    kwargs = dict(benchmark_frame=frames["SPY"], benchmark_entry_cost_bps=entry_cost, account_type="taxable")
    taxed = abi1_driver.evaluate_family(curve, tax_policy="realization", **kwargs)
    untaxed = abi1_driver.evaluate_family(curve, tax_policy="none", **kwargs)
    sens_curve = mod.build_next_open_sensitivity(frames["SPY"], legs, frames["^IRX"], leg_cost_bps=leg_costs,
                                                 spy_entry_cost_bps=entry_cost)
    sens = abi1_driver.evaluate_family(sens_curve, tax_policy="realization", **kwargs)
    bench = benchmark_relative.cost_adjusted_benchmark_daily_returns(frames["SPY"], entry_cost_bps=entry_cost)
    return frames, curve, taxed, untaxed, sens, bench, leg_costs, entry_cost


def _diagnostics(built, *, result=None):
    frames, curve, taxed, untaxed, sens, bench, leg_costs, entry_cost = built
    return driver.overlay_diagnostics(
        curve, taxed if result is None else result, untaxed_result=untaxed, sensitivity_result=sens,
        benchmark_daily=bench, leg_cost_bps=leg_costs, benchmark_entry_cost_bps=entry_cost)


def test_details_equal_independent_aggregates_of_the_trades_frame():
    built = _built()
    curve = built[1]
    diag = _diagnostics(built)
    trades = curve["trades_df"]
    assert diag["n_windows"] == len(trades)
    assert diag["worst_window_net_ret"] == trades["net_ret"].min()
    assert diag["best_window_net_ret"] == trades["net_ret"].max()
    assert diag["sum_window_financing_ret"] == pytest.approx(trades["financing_ret"].sum())
    assert diag["sum_window_cost_ret"] == pytest.approx(trades["cost_ret"].sum())
    assert diag["first_window_entry"] == trades["entry_date"].iloc[0].strftime("%Y-%m-%d")
    assert diag["last_window_exit"] == trades["exit_date"].iloc[-1].strftime("%Y-%m-%d")
    assert diag["overlay_sessions"] == int(curve["overlay_mask"].sum())
    assert diag["overlay_session_share"] == pytest.approx(curve["overlay_mask"].mean())


def test_tax_identity_ratio_is_growth_from_the_entry_close_to_the_exit_close():
    built = _built()
    curve = built[1]
    diag = _diagnostics(built)
    account = curve["strategy_daily"].to_numpy()
    idx = curve["strategy_daily"].index
    ratios = []
    for _, row in curve["trades_df"].iterrows():
        entry, exit_ = idx.get_loc(row["entry_date"]), idx.get_loc(row["exit_date"])
        growth = np.prod(1.0 + account[entry + 1:exit_ + 1])       # sizing close -> exit close
        ratios.append(growth / (1.0 + row["net_ret"]))
    assert diag["tax_identity_ratio_min"] == pytest.approx(min(ratios), rel=1e-12)
    assert diag["tax_identity_ratio_max"] == pytest.approx(max(ratios), rel=1e-12)


def test_scored_beta_is_read_from_the_taxed_result():
    from mining import benchmark_relative
    built = _built()
    taxed, untaxed, bench = built[2], built[3], built[5]
    diag = _diagnostics(built)
    from_taxed = benchmark_relative.ols_alpha_beta(taxed["log_excess"], bench)["beta"]
    from_untaxed = benchmark_relative.ols_alpha_beta(untaxed["log_excess"], bench)["beta"]
    assert from_taxed != from_untaxed                # tax genuinely bites in the fixture
    assert diag["scored_excess_beta_to_spy"] == pytest.approx(from_taxed, rel=1e-12)


def test_the_sensitivity_block_is_read_from_the_next_open_result_not_the_scored_one():
    built = _built()
    taxed, sens = built[2], built[4]
    diag = _diagnostics(built)
    assert diag["next_open_sensitivity"]["sharpe_daily_excess"] == pytest.approx(
        sens["layer1"]["sharpe_daily_excess"], rel=1e-12)
    assert diag["next_open_sensitivity"]["sharpe_daily_excess"] != pytest.approx(
        taxed["layer1"]["sharpe_daily_excess"], rel=1e-9)


# ------------------------------- opt-in history end
#
# --history-end only DROPS sessions after the cutoff, in memory, after the OHLC
# cache write. The default (no setting) is the behaviour the tests above pin;
# everything below covers the setting.

from mining import history_cutoff  # noqa: E402

_ALL_SERIES = ("IEF", "TLT", "AGG", "SPY", "^IRX")
# The production-shaped fixtures end many sessions after any window exit, so a
# real run with a history end drops far more than the default real-run limit
# (history_cutoff.MAX_UNACCEPTED_DROP_SESSIONS) from every series: a real run
# that is not about the guard says explicitly that it accepts the drop.
_ACCEPT_ANY_DROP = 1_000_000


def _d(stamp):
    return pd.Timestamp(stamp).strftime("%Y-%m-%d")


def _run_he(tmp_path, history_end, *, dry_run=True, frames=None, run_id="tmo-he", seed_ledger=False,
            tested_date="2026-09-23", max_history_drop=None):
    """compute_and_record with a history end; ``seed_ledger`` first writes an
    unrelated outcome so a test can prove the ledger file is byte-identical.
    ``max_history_drop`` None leaves the keyword out (the driver's own default)."""
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path, synthetic=dry_run)
    if seed_ledger:
        trial_ledger.record_outcome(ledger_path, "unrelated_family", "earlier", "2026-09-01",
                                    stage="cut_dsr", dsr_value=0.1, sharpe_in_sample=0.0,
                                    construction="ABI-1", note="prior")
    extra = {} if max_history_drop is None else {"max_history_drop": max_history_drop}
    result = driver.compute_and_record(
        catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path,
        run_id=run_id, tested_date=tested_date, dry_run=dry_run,
        signal_source_path="scripts/run_treasury_month_end_overlay_abi1.py",
        signal_source_sha256="a" * 64, research_root=research_root,
        ohlc_loader=_loader(frames or _frames()), history_end=history_end, **extra)
    return result, ledger_path, research_root


def _window_exit(window=-15):
    """Hole-free production-shaped frames plus one window's entry and scheduled
    exit (both inside the data, well before its last bar)."""
    from mining.batch5 import treasury_month_end_index_demand_overlay as mod
    frames = _frames()
    leg_index = frames["IEF"].index
    entry, exit_ = mod.windows_in_data(leg_index)["windows"][window]
    return frames, leg_index, entry, exit_


def _without_session(frames, gone, symbols=_ALL_SERIES):
    return {s: (f.drop(gone) if s in symbols else f) for s, f in frames.items()}


def _snapshot(path):
    with open(path, "rb") as fh:
        return fh.read()


def test_history_end_off_adds_nothing_to_the_result_or_the_note(tmp_path):
    omitted, _ = _run(_mk(tmp_path, "omitted"), run_id="tmo-he")
    explicit, _, _ = _run_he(_mk(tmp_path, "explicit"), None)
    for out in (omitted, explicit):
        assert "history_end" not in out and "raw_last_bar_dates" not in out
        assert "history_end_requested" not in out and "history_end_guard_would_refuse" not in out
        assert history_cutoff.HISTORY_END_MARKER not in out["cost_model_note"]
    assert json.dumps(omitted, default=str, sort_keys=True) == json.dumps(explicit, default=str, sort_keys=True)


@pytest.mark.parametrize("dropped_from,code", [(_ALL_SERIES, "missing_exchange_session"),
                                               (("IEF", "AGG"), "calendar_mismatch")])
def test_an_interior_vendor_hole_refuses_by_default_and_a_history_end_before_it_scores_a_dry_run(
        tmp_path, dropped_from, code):
    frames, leg_index, _, exit_ = _window_exit()
    gone = leg_index[leg_index.get_loc(exit_) + 1]           # the session right after a window's exit
    holed = _without_session(frames, gone, dropped_from)
    refused, _, _ = _run_he(_mk(tmp_path, "off"), None, frames=holed)
    assert refused["reason_code"] == code and "verdict" not in refused
    result, ledger_path, _ = _run_he(_mk(tmp_path, "on"), _d(exit_), frames=holed)
    assert "reason_code" not in result and result["verdict"] in (
        "not_scoreable", "cut_dsr", "cut_risk_refusal", "holdout_not_evaluable", "cut_holdout", "survived")
    assert result["account_end"] == _d(exit_)                # the account ends exactly at the cutoff
    assert result["last_bar_dates"] == {s: _d(exit_) for s in _ALL_SERIES}       # the TRIMMED history
    assert result["raw_last_bar_dates"] == {s: _d(frames[s].index[-1]) for s in _ALL_SERIES}
    assert result["overlay_diagnostics"]["final_window_truncated"] is False
    assert result["overlay_diagnostics"]["last_window_exit"] == _d(exit_)
    assert not os.path.exists(ledger_path)                   # a dry run never writes


def test_the_real_vendor_shape_two_series_missing_their_last_bars_scores_with_a_history_end(tmp_path):
    """IEF and AGG lack the last bar TLT and SPY have (a vendor data shape seen
    in practice): refused by default, scoreable when the history ends the session before."""
    frames = _frames()
    frames["IEF"], frames["AGG"] = frames["IEF"].iloc[:-1], frames["AGG"].iloc[:-1]
    refused, _, _ = _run_he(_mk(tmp_path, "off"), None, frames=frames)
    assert refused["reason_code"] == "unequal_trailing_coverage"
    cutoff = frames["IEF"].index[-1]                         # the last bar every series has
    result, _, _ = _run_he(_mk(tmp_path, "on"), _d(cutoff), frames=frames)
    assert "reason_code" not in result and result["account_end"] == _d(cutoff)
    assert result["raw_last_bar_dates"]["TLT"] == _d(frames["TLT"].index[-1]) != _d(cutoff)
    assert result["raw_last_bar_dates"]["IEF"] == _d(cutoff)


def test_a_trimmed_run_scores_exactly_the_prefix_of_an_untrimmed_run(tmp_path, monkeypatch):
    frames, _, _, exit_ = _window_exit()
    seen = []
    real = driver.abi1_driver.evaluate_family

    def spy(curve, **kwargs):
        seen.append(curve)
        return real(curve, **kwargs)
    monkeypatch.setattr(driver.abi1_driver, "evaluate_family", spy)
    _run_he(_mk(tmp_path, "full"), None, frames=frames)
    full = seen[0]                                           # the first evaluation is the scored one
    seen.clear()
    _run_he(_mk(tmp_path, "trim"), _d(exit_), frames=frames)
    trimmed = seen[0]
    cut = pd.Timestamp(exit_)
    assert full["strategy_daily"].index[-1] > cut            # the untrimmed run really has more
    assert trimmed["strategy_daily"].index[-1] == cut
    for key in ("strategy_daily", "overlay_mask", "financing_rate"):
        expected = full[key].loc[:cut]
        assert trimmed[key].index.equals(expected.index), key
        np.testing.assert_array_equal(trimmed[key].to_numpy(), expected.to_numpy(), err_msg=key)
    pd.testing.assert_frame_equal(
        trimmed["trades_df"], full["trades_df"][full["trades_df"]["exit_date"] <= cut].reset_index(drop=True))


def test_bars_after_the_history_end_cannot_reach_any_scored_number(tmp_path):
    """Every later use of every frame variable must see the trimmed history: the
    post-cutoff bars are replaced with wildly different (finite, positive) ones
    -- each row scaled by its own random factor between 0.7 and 1.4, so daily
    moves of tens of percent, yet still a valid account (a wider band makes the
    untrimmed control itself an invalid account) -- and the WHOLE result is
    unchanged, while the same poison changes an untrimmed run (so it really is
    loud enough to matter)."""
    frames, _, _, exit_ = _window_exit()
    rng = np.random.RandomState(99)
    poisoned = {}
    for symbol, frame in frames.items():
        bad = frame.copy()
        after = np.asarray(bad.index > exit_)
        bad.loc[after, :] = bad.loc[after].to_numpy() * rng.uniform(0.7, 1.4, size=(int(after.sum()), 1))
        poisoned[symbol] = bad
    clean, _, _ = _run_he(_mk(tmp_path, "clean"), _d(exit_), frames=frames)
    dirty, _, _ = _run_he(_mk(tmp_path, "dirty"), _d(exit_), frames=poisoned)
    assert "reason_code" not in clean
    assert json.dumps(clean, default=str, sort_keys=True) == json.dumps(dirty, default=str, sort_keys=True)
    plain_clean, _, _ = _run_he(_mk(tmp_path, "plain-clean"), None, frames=frames)
    plain_dirty, _, _ = _run_he(_mk(tmp_path, "plain-dirty"), None, frames=poisoned)
    # (the poison sits in the tail, which is the ABI-1 holdout: the in-sample
    # Sharpe alone would not move, the rest of the report does)
    assert json.dumps(plain_clean, default=str, sort_keys=True) != json.dumps(plain_dirty, default=str, sort_keys=True)
    assert plain_clean["overlay_diagnostics"]["worst_window_net_ret"] != plain_dirty["overlay_diagnostics"]["worst_window_net_ret"]


def test_a_real_run_with_a_history_end_records_the_clause_and_provenance_and_pins_the_holdout_end(tmp_path):
    frames, leg_index, _, exit_ = _window_exit()
    cutoff = _d(exit_)
    gone = leg_index[leg_index.get_loc(exit_) + 1]
    holed = _without_session(frames, gone)                   # a hole an untrimmed real run refuses
    result, ledger_path, _ = _run_he(tmp_path, cutoff, dry_run=False, frames=holed,
                                     max_history_drop=_ACCEPT_ANY_DROP)
    assert result["stage"] in ("survived", "cut_dsr", "cut_holdout", "cut_risk_refusal")
    assert result["history_end_requested"] == cutoff         # the success payload names the raw setting too
    assert "history_end_guard_would_refuse" not in result    # a real run that passed the guard has nothing to predict
    prov = result["history_end"]
    assert prov["history_end"] == cutoff and prov["opt_in"] is True
    assert prov["decision"] == history_cutoff.DECISION
    assert sorted(prov["by_symbol"]) == sorted(_ALL_SERIES)
    raw_last = _d(frames["IEF"].index[-1])
    assert all(rec["raw_last"] == raw_last and rec["kept_last"] == cutoff for rec in prov["by_symbol"].values())
    assert raw_last in prov["dropped_dates"] and _d(gone) not in prov["dropped_dates"]
    assert result["raw_last_bar_dates"] == {s: raw_last for s in _ALL_SERIES}
    assert result["last_bar_dates"] == {s: cutoff for s in _ALL_SERIES}
    assert result["account_end"] == cutoff
    json.dumps(result, allow_nan=False, default=str)
    rows = trial_ledger.read_all(ledger_path)
    assert sorted(r.get("event") for r in rows) == ["first_test", "outcome"]
    note = [r for r in rows if r.get("event") == "outcome"][0]["note"]
    clause = history_cutoff.history_end_clause(prov)
    assert history_cutoff.HISTORY_END_MARKER in note and clause in note
    assert result["cost_model_note"] == note.split("; ", 3)[3]             # the report carries the same text
    assert f"ended at {cutoff}" in note and raw_last in note and f"permanently ends at {cutoff}" in note
    assert f"through {cutoff} (legs) / {cutoff} (SPY)" in note             # the price-data sentence is the trimmed history
    # the ABI-1 holdout is the tail of the scored account: its end is pinned to the cutoff
    first_test = [r for r in rows if r.get("event") == "first_test"][0]
    assert first_test["holdout_end"] == cutoff


def test_an_unscoreable_real_run_with_a_history_end_still_pins_the_cutoff_in_the_ledger_row(tmp_path):
    """The two unscored verdicts carry the history-end clause as their note (the
    driver hands it over as ``unscored_note``; the tests at the end of this file
    pin that). This test pins the other half: the outcome row's evidence still
    names the cutoff as the account end and the holdout end, and the result names
    the setting."""
    from mining.batch5 import treasury_month_end_index_demand_overlay as mod
    frames = _frames()
    exit_ = mod.windows_in_data(frames["IEF"].index)["windows"][8][1]      # about 9 months of account
    result, ledger_path, _ = _run_he(tmp_path, _d(exit_), dry_run=False, frames=frames,
                                     max_history_drop=_ACCEPT_ANY_DROP)
    assert result["verdict"] == "not_scoreable" and result["stage"] == "insufficient_evidence"
    assert result["history_end"]["history_end"] == _d(exit_) and result["account_end"] == _d(exit_)
    outcomes = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    assert len(outcomes) == 1
    assert outcomes[0]["evidence"]["observed"]["account_end"] == _d(exit_)
    assert outcomes[0]["evidence"]["holdout"]["holdout_end"] == _d(exit_)


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("value,code", [
    ("2015/06/30", "history_end_invalid"), ("", "history_end_invalid"),
    ("2015" + "0630", "history_end_invalid"),                # the compact form is not the setting's format
    ("2015-6-30", "history_end_invalid"),
    ("2015-02-30", "history_end_invalid"), (" 2015-06-30", "history_end_invalid"),
    ("2015-06-27", "history_end_not_a_session"),             # a Saturday
    ("2015-12-25", "history_end_not_a_session"),             # an exchange holiday
], ids=["slashes", "empty", "compact", "unpadded", "impossible_day", "leading_space", "saturday", "holiday"])
def test_an_unusable_history_end_is_refused_before_anything_is_loaded_or_written(tmp_path, dry_run, value, code):
    result, ledger_path, research_root = _run_he(tmp_path, value, dry_run=dry_run, seed_ledger=True)
    seeded = _snapshot(ledger_path)
    assert result["reason_code"] == code and "verdict" not in result
    assert result["history_end_requested"] == value and result["family_id"] == _FAMILY
    assert result["reason"]
    json.dumps(result, allow_nan=False)
    assert not os.path.exists(research_root)                 # not even the OHLC cache: refused before any load
    assert _snapshot(ledger_path) == seeded


@pytest.mark.parametrize("dry_run", [True, False])
def test_a_history_end_no_series_covers_is_refused_and_no_ledger_row_is_written(tmp_path, dry_run):
    result, ledger_path, _ = _run_he(tmp_path, "2017-03-15", dry_run=dry_run, seed_ledger=True)   # a real session after the data
    assert result["reason_code"] == "history_end_not_covered" and "verdict" not in result
    assert result["symbols"] == sorted(_ALL_SERIES) and "offending_symbols" not in result
    assert "2017-03-15" in result["reason"]
    assert len(trial_ledger.read_all(ledger_path)) == 1        # only the seeded unrelated row


@pytest.mark.parametrize("lacking", ["IEF", "SPY", "^IRX"])
def test_a_history_end_one_series_has_no_bar_on_is_refused_naming_that_series(tmp_path, lacking):
    """EVERY loaded series must carry a bar on the cutoff -- including the bill
    series (a bond-market holiday would otherwise leave it silently uncovered)."""
    frames, _, _, exit_ = _window_exit()
    frames[lacking] = frames[lacking].drop(exit_)
    result, ledger_path, _ = _run_he(tmp_path, _d(exit_), dry_run=False, frames=frames, seed_ledger=True)
    assert result["reason_code"] == "history_end_not_covered"
    assert result["symbols"] == [lacking]
    assert len(trial_ledger.read_all(ledger_path)) == 1


def test_the_truncated_final_window_refusal_still_fires_on_a_trimmed_history(tmp_path):
    """The untrimmed data holds the window's exit, so the refusal here can only
    come from running the existing check on the TRIMMED frames."""
    frames, leg_index, entry, exit_ = _window_exit()
    inside = leg_index[leg_index.get_loc(entry) + 1]         # T-2: the trade would be closed early
    assert entry < inside < exit_ and frames["IEF"].index[-1] > exit_
    dry, _, _ = _run_he(_mk(tmp_path, "dry"), _d(inside), frames=frames)
    assert dry["overlay_diagnostics"]["final_window_truncated"] is True and dry["account_end"] == _d(inside)
    real, ledger_path, _ = _run_he(_mk(tmp_path, "real"), _d(inside), dry_run=False, frames=frames, run_id="tmo-he-2",
                                   max_history_drop=_ACCEPT_ANY_DROP)
    assert real["reason_code"] == "truncated_final_window" and "verdict" not in real
    assert not os.path.exists(ledger_path)
    complete, ledger_path, _ = _run_he(_mk(tmp_path, "ok"), _d(exit_), dry_run=False, frames=frames, run_id="tmo-he-3",
                                       max_history_drop=_ACCEPT_ANY_DROP)
    assert "reason_code" not in complete and len(trial_ledger.read_all(ledger_path)) == 2


def test_a_trimmed_run_leaves_the_full_history_in_the_ohlc_cache_and_a_real_run_still_refuses_that_cache(tmp_path):
    """The trim happens after the cache write and never inside the loader: the
    cached CSVs keep the whole vendor history, so a later run cannot inherit a
    silently shortened one, and the stale-cache refusal is untouched."""
    frames, _, _, exit_ = _window_exit()
    dry, _, research_root = _run_he(tmp_path, _d(exit_), frames=frames, run_id="tmo-cache")
    assert set(dry["ohlc_sources"].values()) == {"fetched"}
    cache_dir = os.path.join(research_root, "tmo-cache", "artifacts", "ohlc")
    for symbol, frame in frames.items():
        cached = pd.read_csv(os.path.join(cache_dir, driver.batch5._cache_name(symbol)))
        assert len(cached) == len(frame), symbol
        assert cached["Date"].iloc[-1] == _d(frame.index[-1]) > _d(exit_), symbol
    # with or without the setting, even one no cached series covers: the same
    # refusal, so the trim runs AFTER the stale-cache check, never before it
    for history_end in (None, _d(exit_), "2017-03-15"):
        real, ledger_path, _ = _run_he(tmp_path, history_end, dry_run=False, frames=frames, run_id="tmo-cache")
        assert real["reason_code"] == "stale_ohlc_cache" and real["cached_symbols"] == list(_ALL_SERIES)
        assert "history_end" not in real and not os.path.exists(ledger_path)


# ------------------------------------------------------------ history end: cli

def test_cli_history_end_dry_run_records_the_setting_in_the_result_file(tmp_path, monkeypatch, capsys):
    frames, _, _, exit_ = _window_exit()
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path)
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(frames))
    code = driver.main(["--quotes", quotes_path, "--dry-run", "--catalog", catalog_path,
                        "--ledger", ledger_path, "--research-root", research_root,
                        "--run-id", "cli-he", "--history-end", _d(exit_)])
    assert code == 0
    with open(os.path.join(research_root, "cli-he", "treasury-month-end-overlay-dry-run.json"),
              "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["history_end_requested"] == _d(exit_)
    result = payload["result"]
    assert result["history_end"]["history_end"] == _d(exit_) and result["account_end"] == _d(exit_)
    assert result["history_end_requested"] == _d(exit_)      # the same key name as the top-level payload's
    assert history_cutoff.HISTORY_END_MARKER in result["cost_model_note"]
    assert not os.path.exists(ledger_path)
    assert json.loads(capsys.readouterr().out)["result"]["history_end"]["opt_in"] is True


def test_cli_without_the_setting_writes_a_payload_with_no_history_end_keys(tmp_path, monkeypatch):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path)
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(_frames()))
    assert driver.main(["--quotes", quotes_path, "--dry-run", "--catalog", catalog_path,
                        "--ledger", ledger_path, "--research-root", research_root, "--run-id", "cli-off"]) == 0
    with open(os.path.join(research_root, "cli-off", "treasury-month-end-overlay-dry-run.json"),
              "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert "history_end_requested" not in payload and "history_end" not in payload["result"]


@pytest.mark.parametrize("value,code,cache_exists", [
    ("2015-06-27", "history_end_not_a_session", False),
    ("", "history_end_invalid", False),
    ("2017-03-15", "history_end_not_covered", True),          # refused after the (untouched) cache write
])
def test_cli_refused_history_end_writes_no_result_file_and_exits_nonzero(
        tmp_path, monkeypatch, capsys, value, code, cache_exists):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path)
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(_frames()))
    exit_code = driver.main(["--quotes", quotes_path, "--dry-run", "--catalog", catalog_path,
                             "--ledger", ledger_path, "--research-root", research_root,
                             "--run-id", "cli-refused", "--history-end", value])
    assert exit_code == 2
    printed = json.loads(capsys.readouterr().out)
    assert printed["output"] is None and printed["result"]["reason_code"] == code
    run_dir = os.path.join(research_root, "cli-refused")
    assert os.path.exists(run_dir) is cache_exists
    if cache_exists:
        assert [n for n in os.listdir(run_dir) if n.endswith(".json")] == []   # no result file
    assert not os.path.exists(ledger_path)


def test_the_cli_help_states_the_setting_is_opt_in_and_permanent(capsys):
    with pytest.raises(SystemExit):
        driver.main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "--history-end" in text and "opt-in" in text and "opt-in history-end setting" in text
    assert "permanent" in text


# ------------------------- history end: the clause on the UNSCORED ledger rows
#
# record_abi1_outcome writes extra_note only on the four SCORED verdicts. A run
# that ends unscored still spends the family's one permanent ABI-1 slot, so with
# a history end the driver ALSO hands the clause to the recorder's separate
# unscored_note keyword. With no history end the keyword is not passed at all.

def _spy_record_abi1_outcome(monkeypatch):
    """Record the keyword arguments of every record_abi1_outcome call and still
    run the real function (so a tmp ledger row is really written)."""
    calls = []
    real = driver.abi1_driver.record_abi1_outcome

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)
    monkeypatch.setattr(driver.abi1_driver, "record_abi1_outcome", spy)
    return calls


def _short_account_frames_and_cutoff():
    """Production-shaped frames plus a window exit about nine months into the
    account: a history ended there is too short to score (not_scoreable)."""
    from mining.batch5 import treasury_month_end_index_demand_overlay as mod
    frames = _frames()
    return frames, mod.windows_in_data(frames["IEF"].index)["windows"][8][1]


def _scored_frames_and_cutoff():
    """A history ended at a window exit deep enough to score (the shape the
    scored-row test above uses, with the same kind of vendor hole)."""
    frames, leg_index, _, exit_ = _window_exit()
    gone = leg_index[leg_index.get_loc(exit_) + 1]
    return _without_session(frames, gone), exit_


@pytest.mark.parametrize("world,expected_stages", [
    (_short_account_frames_and_cutoff, ("insufficient_evidence",)),
    (_scored_frames_and_cutoff, ("survived", "cut_dsr", "cut_holdout", "cut_risk_refusal")),
])
def test_a_real_run_with_a_history_end_hands_the_clause_to_the_recorders_unscored_note(
        tmp_path, monkeypatch, world, expected_stages):
    frames, cutoff = world()
    calls = _spy_record_abi1_outcome(monkeypatch)
    result, _, _ = _run_he(tmp_path, _d(cutoff), dry_run=False, frames=frames,
                           max_history_drop=_ACCEPT_ANY_DROP)
    assert result["stage"] in expected_stages
    assert len(calls) == 1
    kwargs = calls[0]
    clause = history_cutoff.history_end_clause(result["history_end"])
    assert kwargs["unscored_note"] == clause
    assert kwargs["unscored_note"].startswith(history_cutoff.HISTORY_END_MARKER)
    # the scored-row path carries the same clause in extra_note, and it LEADS
    # the note (like the gold and commodity drivers), once
    assert kwargs["extra_note"].startswith(f"{clause} cost model: ")
    assert kwargs["extra_note"].count(clause) == 1


def test_a_real_run_without_a_history_end_does_not_pass_unscored_note_at_all(tmp_path, monkeypatch):
    calls = _spy_record_abi1_outcome(monkeypatch)
    off, _, _ = _run_he(_mk(tmp_path, "off"), None, dry_run=False, frames=_frames(n_legs=220), run_id="tmo-off")
    assert off["verdict"] == "not_scoreable" and len(calls) == 1
    assert "unscored_note" not in calls[0]                   # not even unscored_note=None
    assert history_cutoff.HISTORY_END_MARKER not in calls[0]["extra_note"]
    # ... and the only difference a history end makes to the call is that one keyword
    frames, cutoff = _short_account_frames_and_cutoff()
    _run_he(_mk(tmp_path, "on"), _d(cutoff), dry_run=False, frames=frames, run_id="tmo-on",
            max_history_drop=_ACCEPT_ANY_DROP)
    assert len(calls) == 2
    assert set(calls[1]) - set(calls[0]) == {"unscored_note"} and set(calls[0]) <= set(calls[1])


def test_an_unscoreable_real_run_with_a_history_end_writes_the_clause_as_the_ledger_rows_note(tmp_path):
    """The real compute path, a tmp ledger, no fake: the written
    insufficient-evidence row carries the marker, the cutoff and the raw last
    bars in its note, verbatim; the same kind of run with no history end still
    writes no note at all."""
    frames, cutoff = _short_account_frames_and_cutoff()
    result, ledger_path, _ = _run_he(_mk(tmp_path, "on"), _d(cutoff), dry_run=False, frames=frames,
                                     run_id="tmo-on", max_history_drop=_ACCEPT_ANY_DROP)
    assert result["verdict"] == "not_scoreable" and result["stage"] == "insufficient_evidence"
    outcomes =[r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    assert len(outcomes) == 1 and outcomes[0]["stage"] == "insufficient_evidence"
    note = outcomes[0]["note"]
    assert note == history_cutoff.history_end_clause(result["history_end"])       # verbatim, nothing added or dropped
    assert history_cutoff.HISTORY_END_MARKER in note
    raw_last = _d(frames["IEF"].index[-1])
    assert raw_last > _d(cutoff)                             # the cutoff really dropped later sessions
    assert f"ended at {_d(cutoff)}" in note and f"permanently ends at {_d(cutoff)}" in note
    assert raw_last in note and "The setting added, moved or repaired no bar" in note
    assert "Nothing was filled or patched" not in note and "not by data availability" not in note
    # the row still pins the account end and the holdout end at the cutoff
    assert outcomes[0]["evidence"]["observed"]["account_end"] == _d(cutoff)
    assert outcomes[0]["evidence"]["holdout"]["holdout_end"] == _d(cutoff)
    # control: an unscored run with no history end writes no note
    off, off_ledger, _ = _run_he(_mk(tmp_path, "off"), None, dry_run=False, frames=_frames(n_legs=220),
                                 run_id="tmo-off")
    assert off["verdict"] == "not_scoreable"
    off_outcomes = [r for r in trial_ledger.read_all(off_ledger) if r.get("event") == "outcome"]
    assert len(off_outcomes) == 1 and off_outcomes[0]["note"] is None


# ---------------- history end: the real-run guard, one refusal shape, the door
#
# A real run permanently spends the family's one-shot ABI-1 slot, so a history
# end that is later than the run's tested date, or that drops more sessions from
# some series than --max-history-drop allows, is refused on a real run (nothing
# reaches the ledger). A dry run is never refused by the guard: it reports the
# message a real run would refuse with, so a rehearsal predicts the real run.

_REFUSAL_KEYS = ["reason_code", "reason", "history_end_requested", "symbols", "family_id"]


def _before(stamp, days=1):
    return _d(pd.Timestamp(stamp) - pd.Timedelta(days=days))


def test_a_real_run_refuses_a_history_end_that_drops_more_sessions_than_the_limit(tmp_path):
    frames, _, _, exit_ = _window_exit()
    dry, _, _ = _run_he(_mk(tmp_path, "dry"), _d(exit_), frames=frames, run_id="tmo-rehearse")
    drop = dry["history_end"]["by_symbol"]["IEF"]["dropped_sessions"]
    assert drop > history_cutoff.MAX_UNACCEPTED_DROP_SESSIONS       # the fixture really exceeds the default limit
    real, ledger_path, research_root = _run_he(_mk(tmp_path, "real"), _d(exit_), dry_run=False, frames=frames,
                                               seed_ledger=True)
    assert real["reason_code"] == "history_end_drop_too_large" and "verdict" not in real
    assert list(real) == _REFUSAL_KEYS and real["family_id"] == _FAMILY
    assert real["history_end_requested"] == _d(exit_) and real["symbols"] == sorted(_ALL_SERIES)
    assert f"{drop} sessions" in real["reason"] and "pass --max-history-drop N to accept a larger drop" in real["reason"]
    assert len(trial_ledger.read_all(ledger_path)) == 1             # only the seeded unrelated row
    json.dumps(real, allow_nan=False)


def test_the_drop_limit_is_inclusive_and_the_refusal_names_the_worst_series_and_every_offender(tmp_path):
    frames, leg_index, _, exit_ = _window_exit()
    holed = _without_session(frames, leg_index[leg_index.get_loc(exit_) + 1], symbols=("IEF",))
    dry, _, _ = _run_he(_mk(tmp_path, "dry"), _d(exit_), frames=holed, run_id="tmo-rehearse")
    by_symbol = dry["history_end"]["by_symbol"]
    worst = by_symbol["AGG"]["dropped_sessions"]
    assert by_symbol["IEF"]["dropped_sessions"] == worst - 1                     # IEF lacks one of the later sessions
    assert {s: r["dropped_sessions"] for s, r in by_symbol.items() if s != "IEF"} == {
        s: worst for s in _ALL_SERIES if s != "IEF"}
    refused, ledger_path, _ = _run_he(_mk(tmp_path, "refused"), _d(exit_), dry_run=False, frames=holed,
                                      max_history_drop=worst - 1, seed_ledger=True)
    assert refused["reason_code"] == "history_end_drop_too_large"
    assert refused["symbols"] == ["AGG", "SPY", "TLT", "^IRX"]                    # every offender, sorted; IEF is at the limit
    assert f"{worst} sessions from AGG" in refused["reason"]                      # ties go to the first symbol
    assert f"more than the {worst - 1} a real run accepts" in refused["reason"]
    assert len(trial_ledger.read_all(ledger_path)) == 1
    accepted, ledger_path, _ = _run_he(_mk(tmp_path, "accepted"), _d(exit_), dry_run=False, frames=holed,
                                       max_history_drop=worst)                   # the limit itself is accepted
    assert "reason_code" not in accepted and accepted["history_end"]["history_end"] == _d(exit_)
    assert len(trial_ledger.read_all(ledger_path)) == 2                           # first_test + outcome


def test_a_real_run_refuses_a_history_end_later_than_its_tested_date_before_it_looks_at_the_drop(tmp_path):
    frames, _, _, exit_ = _window_exit()
    tested = _before(exit_)
    # the default limit is exceeded too: the tested-date refusal comes first
    real, ledger_path, _ = _run_he(_mk(tmp_path, "real"), _d(exit_), dry_run=False, frames=frames,
                                   tested_date=tested, seed_ledger=True)
    assert real["reason_code"] == "history_end_after_tested_date" and "verdict" not in real
    assert list(real) == _REFUSAL_KEYS and real["symbols"] == [] and real["family_id"] == _FAMILY
    assert real["history_end_requested"] == _d(exit_)
    assert tested in real["reason"] and _d(exit_) in real["reason"]
    assert len(trial_ledger.read_all(ledger_path)) == 1
    still, _, _ = _run_he(_mk(tmp_path, "still"), _d(exit_), dry_run=False, frames=frames, tested_date=tested,
                          max_history_drop=_ACCEPT_ANY_DROP)
    assert still["reason_code"] == "history_end_after_tested_date"               # accepting the drop does not accept this
    same_day, ledger_path, _ = _run_he(_mk(tmp_path, "same-day"), _d(exit_), dry_run=False, frames=frames,
                                       tested_date=_d(exit_), max_history_drop=_ACCEPT_ANY_DROP)
    assert "reason_code" not in same_day and len(trial_ledger.read_all(ledger_path)) == 2   # equal is fine


@pytest.mark.parametrize("case,code", [("drop", "history_end_drop_too_large"),
                                       ("tested_date", "history_end_after_tested_date")])
def test_a_dry_run_is_never_refused_by_the_guard_but_predicts_the_real_runs_refusal(tmp_path, case, code):
    frames, _, _, exit_ = _window_exit()
    kwargs = ({} if case == "drop"
              else {"tested_date": _before(exit_), "max_history_drop": _ACCEPT_ANY_DROP})
    dry, _, _ = _run_he(_mk(tmp_path, "dry"), _d(exit_), frames=frames, run_id="tmo-a", **kwargs)
    assert "reason_code" not in dry and dry["verdict"]                            # rehearsed all the way through
    predicted = dry["history_end_guard_would_refuse"]
    assert isinstance(predicted, str) and predicted
    real, ledger_path, _ = _run_he(_mk(tmp_path, "real"), _d(exit_), dry_run=False, frames=frames, run_id="tmo-b",
                                   **kwargs)
    # the same words, so the rehearsal predicts it (the real reason adds the cache-kept sentence)
    assert real["reason_code"] == code and real["reason"].startswith(predicted)
    assert "retry with a fresh --run-id" in real["reason"]
    assert not os.path.exists(ledger_path)


def test_a_dry_run_that_the_guard_would_pass_says_so_with_none_and_the_real_run_then_scores(tmp_path):
    frames, _, _, exit_ = _window_exit()
    dry, _, _ = _run_he(_mk(tmp_path, "dry"), _d(exit_), frames=frames, run_id="tmo-a",
                        max_history_drop=_ACCEPT_ANY_DROP)
    assert "history_end_guard_would_refuse" in dry and dry["history_end_guard_would_refuse"] is None
    real, _, _ = _run_he(_mk(tmp_path, "real"), _d(exit_), dry_run=False, frames=frames, run_id="tmo-b",
                         max_history_drop=_ACCEPT_ANY_DROP)
    assert "reason_code" not in real


@pytest.mark.parametrize("bad", [-1, True, 2.5, "3"])
def test_an_unusable_max_history_drop_is_refused_when_the_setting_is_on_and_ignored_when_it_is_off(tmp_path, bad):
    """The value is checked at the very door (``history_cutoff.validate_guard_arguments``),
    on a dry run too (a dry run rehearses the real run), so it can never fail after the
    load step wrote the per-run OHLC cache: nothing is loaded, nothing is written."""
    frames, _, _, exit_ = _window_exit()
    for dry_run, name in ((False, "real"), (True, "dry")):
        refused, ledger_path, research_root = _run_he(
            _mk(tmp_path, name), _d(exit_), dry_run=dry_run, frames=frames, max_history_drop=bad, seed_ledger=True)
        assert refused["reason_code"] == "history_end_invalid" and "max_drop" in refused["reason"], name
        assert "verdict" not in refused and "history_end_guard_would_refuse" not in refused, name
        assert list(refused) == _REFUSAL_KEYS and len(trial_ledger.read_all(ledger_path)) == 1, name
        assert not os.path.exists(research_root), name             # not even the OHLC cache: refused at the door
    off, _, _ = _run_he(_mk(tmp_path, "off"), None, frames=frames, max_history_drop=bad)
    assert "reason_code" not in off and "history_end_guard_would_refuse" not in off


def test_the_default_path_is_not_guarded_at_all(tmp_path):
    """With no history end there is nothing to guard: an ordinary real run on a
    complete history scores whatever max_history_drop says."""
    off, ledger_path, _ = _run_he(_mk(tmp_path, "off"), None, dry_run=False, frames=_frames(n_legs=220),
                                  max_history_drop=0)
    assert off["verdict"] == "not_scoreable" and "history_end" not in off
    assert len(trial_ledger.read_all(ledger_path)) == 2


def test_every_history_end_refusal_has_the_one_standard_shape(tmp_path):
    invalid, _, _ = _run_he(_mk(tmp_path, "invalid"), "2015/06/30")
    session, _, _ = _run_he(_mk(tmp_path, "session"), "2015-06-27")
    covered, _, _ = _run_he(_mk(tmp_path, "covered"), "2017-03-15")            # a real session no series covers
    for refusal, code, requested in ((invalid, "history_end_invalid", "2015/06/30"),
                                      (session, "history_end_not_a_session", "2015-06-27"),
                                      (covered, "history_end_not_covered", "2017-03-15")):
        assert list(refusal) == _REFUSAL_KEYS, code
        assert refusal["reason_code"] == code and refusal["history_end_requested"] == requested
        assert refusal["family_id"] == _FAMILY and isinstance(refusal["symbols"], list)
        json.dumps(refusal, allow_nan=False)
    assert invalid["symbols"] == [] and session["symbols"] == [] and covered["symbols"] == sorted(_ALL_SERIES)


def test_a_refusal_raised_after_the_ohlc_cache_write_says_the_cache_is_kept_and_a_fresh_run_id_is_needed(tmp_path):
    covered, _, research_root = _run_he(tmp_path, "2017-03-15")
    assert covered["reason_code"] == "history_end_not_covered"
    assert "2017-03-15" in covered["reason"]
    assert "The per-run OHLC cache written before this check is kept" in covered["reason"]
    assert "no ledger row, result file or report is written" in covered["reason"]
    assert "retry with a fresh --run-id" in covered["reason"]
    assert os.path.isdir(os.path.join(research_root, "tmo-he", "artifacts", "ohlc"))     # the cache really is there


def _drifted_parameters():
    with open(_ENTRY_PATH, "r", encoding="utf-8") as fh:
        drifted = dict(_catalog_entry_from(fh)["parameters"])
    drifted["window_sessions"] = 4
    return drifted


def _door_run(tmp_path, history_end, obstacle, *, tested_date="2026-09-23", **extra):
    """compute_and_record (dry, a scratch tree) with one thing wrong BEFORE any
    history is loaded: the catalog status, the parameters, an earlier ABI-1
    outcome for the family, or a quotes file that is not there. ``tested_date``
    and any ``extra`` keyword (``max_history_drop``) go to compute_and_record."""
    fixture_kwargs = {"status": "ruled_out"} if obstacle == "catalog_status" else (
        {"parameters": _drifted_parameters()} if obstacle == "parameter_drift" else {})
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path, **fixture_kwargs)
    if obstacle == "already_scored_abi1":
        trial_ledger.record_outcome(ledger_path, _FAMILY, "earlier", "2026-09-01", stage="cut_dsr",
                                    dsr_value=0.1, sharpe_in_sample=0.0, construction="ABI-1", note="prior")
    if obstacle == "unreadable_quotes":
        os.remove(quotes_path)
    return driver.compute_and_record(
        catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path, run_id="tmo-door",
        tested_date=tested_date, dry_run=True, signal_source_path="p", signal_source_sha256="s",
        research_root=research_root, ohlc_loader=_loader(_frames()), history_end=history_end, **extra), research_root


@pytest.mark.parametrize("obstacle", ["catalog_status", "parameter_drift", "already_scored_abi1",
                                      "unreadable_quotes"])
def test_a_malformed_or_non_session_history_end_is_refused_at_the_door_before_every_other_check(tmp_path, obstacle):
    # control: the obstacle really is one, with the setting off or well-formed
    if obstacle == "unreadable_quotes":
        with pytest.raises(OSError):
            _door_run(_mk(tmp_path, "control"), None, obstacle)
    else:
        control, _ = _door_run(_mk(tmp_path, "control"), None, obstacle)
        assert control["reason_code"] == obstacle
        good, _ = _door_run(_mk(tmp_path, "good"), "2015-06-30", obstacle)          # a real session: not the door's business
        assert good["reason_code"] == obstacle
    for name, value, code in (("bad-format", "2015/06/30", "history_end_invalid"),
                              ("saturday", "2015-06-27", "history_end_not_a_session")):
        refused, research_root = _door_run(_mk(tmp_path, name), value, obstacle)
        assert refused["reason_code"] == code and list(refused) == _REFUSAL_KEYS
        assert refused["history_end_requested"] == value
        assert not os.path.exists(research_root)                                    # nothing was loaded either


# ------------------------------------------------ history end: the guard, on the cli

def _cli_real_run(tmp_path, monkeypatch, frames, *extra, run_id="cli-real", tested_date="2026-09-23"):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path, synthetic=False)
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(frames))
    code = driver.main(["--quotes", quotes_path, "--catalog", catalog_path, "--ledger", ledger_path,
                        "--research-root", research_root, "--run-id", run_id, "--tested-date", tested_date, *extra])
    return code, ledger_path, os.path.join(research_root, run_id)


def _json_files(run_dir):
    return [] if not os.path.isdir(run_dir) else [n for n in os.listdir(run_dir) if n.endswith(".json")]


def test_cli_real_run_refuses_a_too_large_drop_with_exit_2_and_writes_no_result_file_or_ledger_row(
        tmp_path, monkeypatch, capsys):
    frames, _, _, exit_ = _window_exit()
    code, ledger_path, run_dir = _cli_real_run(tmp_path, monkeypatch, frames, "--history-end", _d(exit_))
    assert code == 2
    printed = json.loads(capsys.readouterr().out)
    assert printed["output"] is None and printed["result"]["reason_code"] == "history_end_drop_too_large"
    assert list(printed["result"]) == _REFUSAL_KEYS
    assert _json_files(run_dir) == [] and not os.path.exists(ledger_path)
    assert os.path.isdir(os.path.join(run_dir, "artifacts", "ohlc"))                 # the cache the load step wrote is kept


def test_cli_real_run_refuses_a_history_end_after_the_tested_date_with_exit_2(tmp_path, monkeypatch, capsys):
    frames, _, _, exit_ = _window_exit()
    code, ledger_path, run_dir = _cli_real_run(
        tmp_path, monkeypatch, frames, "--history-end", _d(exit_), "--max-history-drop", str(_ACCEPT_ANY_DROP),
        tested_date=_before(exit_))
    assert code == 2
    printed = json.loads(capsys.readouterr().out)
    assert printed["output"] is None and printed["result"]["reason_code"] == "history_end_after_tested_date"
    assert _json_files(run_dir) == [] and not os.path.exists(ledger_path)


def test_cli_max_history_drop_accepts_the_drop_and_the_real_run_scores_and_writes_its_result(
        tmp_path, monkeypatch, capsys):
    frames, _, _, exit_ = _window_exit()
    code, ledger_path, run_dir = _cli_real_run(
        tmp_path, monkeypatch, frames, "--history-end", _d(exit_), "--max-history-drop", str(_ACCEPT_ANY_DROP))
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["output"] and "reason_code" not in printed["result"]
    assert _json_files(run_dir) == ["treasury-month-end-overlay-results.json"]
    with open(os.path.join(run_dir, "treasury-month-end-overlay-results.json"), "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["history_end_requested"] == _d(exit_)
    assert payload["result"]["history_end_requested"] == _d(exit_)
    assert payload["result"]["history_end"]["history_end"] == _d(exit_)
    assert len(trial_ledger.read_all(ledger_path)) == 2


@pytest.mark.parametrize("bad", ["-1", "abc", "1.5", ""])
def test_cli_rejects_an_unusable_max_history_drop_before_anything_runs(tmp_path, monkeypatch, capsys, bad):
    frames, _, _, exit_ = _window_exit()
    with pytest.raises(SystemExit) as raised:
        _cli_real_run(tmp_path, monkeypatch, frames, "--history-end", _d(exit_), "--max-history-drop", bad)
    assert raised.value.code == 2
    assert "--max-history-drop" in capsys.readouterr().err
    assert not (tmp_path / "research").exists() and not (tmp_path / "ledger.jsonl").exists()


def test_cli_dry_run_reports_what_the_guard_would_refuse_and_still_exits_zero(tmp_path, monkeypatch, capsys):
    frames, _, _, exit_ = _window_exit()
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path)
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(frames))
    code = driver.main(["--quotes", quotes_path, "--dry-run", "--catalog", catalog_path, "--ledger", ledger_path,
                        "--research-root", research_root, "--run-id", "cli-would", "--history-end", _d(exit_)])
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["output"] and "drop" in printed["result"]["history_end_guard_would_refuse"]
    assert "--max-history-drop N" in printed["result"]["history_end_guard_would_refuse"]
    assert _json_files(os.path.join(research_root, "cli-would")) == ["treasury-month-end-overlay-dry-run.json"]


@pytest.mark.parametrize("code", history_cutoff.HISTORY_END_REASON_CODES)
def test_cli_exits_2_for_every_history_end_reason_code_and_writes_nothing(tmp_path, monkeypatch, capsys, code):
    monkeypatch.setattr(driver, "compute_and_record",
                        lambda **kwargs: {"family_id": _FAMILY, "reason_code": code, "reason": "r"})
    research_root = str(tmp_path / "research")
    exit_code = driver.main(["--quotes", "unused.json", "--dry-run", "--research-root", research_root,
                             "--run-id", "cli-codes"])
    assert exit_code == 2
    printed = json.loads(capsys.readouterr().out)
    assert printed["output"] is None and printed["result"]["reason_code"] == code
    assert not os.path.exists(research_root)


def test_cli_still_exits_zero_and_writes_its_file_for_a_refusal_that_is_not_about_the_history_end(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(driver, "compute_and_record",
                        lambda **kwargs: {"family_id": _FAMILY, "reason_code": "stale_ohlc_cache", "reason": "r"})
    research_root = str(tmp_path / "research")
    assert driver.main(["--quotes", "unused.json", "--dry-run", "--research-root", research_root,
                        "--run-id", "cli-other"]) == 0
    assert _json_files(os.path.join(research_root, "cli-other")) == ["treasury-month-end-overlay-dry-run.json"]


def test_the_cli_help_and_docstring_state_the_guard_and_what_a_refusal_writes(
        capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "100000")                  # argparse must not break "--run-id" at its hyphen
    with pytest.raises(SystemExit):
        driver.main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "--max-history-drop" in text and "--tested-date" in text and "no ledger row" in text
    assert "OHLC cache" in text and "fresh --run-id" in text
    doc = " ".join(driver.__doc__.split())
    assert "history_end_after_tested_date" in doc
    assert "history_end_drop_too_large" in doc and "history_end_guard_would_refuse" in doc
    assert "record_first_test" in doc and "a not_scoreable row pins none" in doc


# -------- history end: the door also validates the guard's two inputs
#
# tested_date and max_history_drop feed the real-run guard, which runs AFTER the load
# step wrote the per-run OHLC cache. A malformed value must therefore be refused at
# the very door (history_cutoff.validate_guard_arguments), on dry runs too (a dry run
# rehearses the real run), so it can never fail after a load: nothing is loaded,
# nothing is written, and there is no cache to retry around. With the setting off the
# two inputs are not looked at (tested_date is only ever written to a permanent row
# by a real run).

_DOOR_CUTOFF = "2015-06-30"                 # a real NYSE session inside the fixtures' history
_BAD_TESTED_DATES = ["2026-9-25", "2026/09/25", "", "2026" + "0925", " 2026-09-25", "not-a-date"]


def _no_load(monkeypatch):
    """Any attempt to load a history is a failure of the test that installs this."""
    def boom(*args, **kwargs):
        raise AssertionError("a history was loaded although the door should have refused first")
    monkeypatch.setattr(driver.batch5, "load_frame", boom)


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("bad", _BAD_TESTED_DATES,
                         ids=["unpadded", "slashes", "empty", "compact", "leading_space", "word"])
def test_a_malformed_tested_date_with_a_valid_history_end_is_refused_at_the_door(
        tmp_path, monkeypatch, dry_run, bad):
    _no_load(monkeypatch)
    result, ledger_path, research_root = _run_he(_mk(tmp_path, "run"), _DOOR_CUTOFF, dry_run=dry_run,
                                                 tested_date=bad, seed_ledger=True)
    assert result["reason_code"] == "history_end_invalid" and "verdict" not in result
    assert list(result) == _REFUSAL_KEYS and result["family_id"] == _FAMILY
    assert result["history_end_requested"] == _DOOR_CUTOFF and result["symbols"] == []
    assert "tested_date" in result["reason"] and repr(bad) in result["reason"]
    assert "OHLC cache" not in result["reason"]              # a door refusal: no cache exists, so no cache-kept sentence
    json.dumps(result, allow_nan=False)
    assert not os.path.exists(research_root)                 # not even the OHLC cache
    assert len(trial_ledger.read_all(ledger_path)) == 1      # only the seeded unrelated row


@pytest.mark.parametrize("obstacle", ["catalog_status", "parameter_drift", "already_scored_abi1",
                                      "unreadable_quotes"])
@pytest.mark.parametrize("which", ["tested_date", "max_history_drop"])
def test_the_guards_two_inputs_are_checked_at_the_door_before_every_other_check(tmp_path, obstacle, which):
    bad = {"tested_date": "2026-9-25", "max_history_drop": -1}
    if obstacle == "unreadable_quotes":
        with pytest.raises(OSError):
            _door_run(_mk(tmp_path, "control"), _DOOR_CUTOFF, obstacle)        # control: the obstacle is real
    else:
        control, _ = _door_run(_mk(tmp_path, "control"), _DOOR_CUTOFF, obstacle)
        assert control["reason_code"] == obstacle
    refused, research_root = _door_run(_mk(tmp_path, "bad"), _DOOR_CUTOFF, obstacle, **{which: bad[which]})
    assert refused["reason_code"] == "history_end_invalid" and list(refused) == _REFUSAL_KEYS
    assert refused["history_end_requested"] == _DOOR_CUTOFF
    assert ("tested_date" if which == "tested_date" else "max_drop") in refused["reason"]
    assert not os.path.exists(research_root)                 # nothing was loaded either


@pytest.mark.parametrize("bad", ["2026-9-25", "not-a-date"])
def test_a_malformed_tested_date_with_no_history_end_gives_the_valid_date_result(tmp_path, monkeypatch, bad):
    """Setting off: tested_date is not looked at by a dry run (a real run only writes it
    into the permanent row), so the result is the valid-date run's, key for key."""
    good, _, _ = _run_he(_mk(tmp_path, "good"), None, tested_date="2026-09-23")
    odd, _, _ = _run_he(_mk(tmp_path, "odd"), None, tested_date=bad)
    assert "reason_code" not in odd and odd["verdict"] == good["verdict"]
    assert "history_end_requested" not in odd and "history_end" not in odd
    assert json.dumps(odd, default=str, sort_keys=True) == json.dumps(good, default=str, sort_keys=True)


@pytest.mark.parametrize("dry_run", [True, False])
def test_cli_a_malformed_tested_date_with_a_history_end_is_refused_at_the_door_with_exit_2(
        tmp_path, monkeypatch, capsys, dry_run):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path, synthetic=dry_run)
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(_frames()))
    _no_load(monkeypatch)
    argv = ["--quotes", quotes_path, "--catalog", catalog_path, "--ledger", ledger_path,
            "--research-root", research_root, "--run-id", "cli-door", "--tested-date", "2026-9-25",
            "--history-end", _DOOR_CUTOFF]
    assert driver.main(argv + (["--dry-run"] if dry_run else [])) == 2
    printed = json.loads(capsys.readouterr().out)
    assert printed["output"] is None and printed["result"]["reason_code"] == "history_end_invalid"
    assert "tested_date" in printed["result"]["reason"]
    assert not os.path.exists(research_root) and not os.path.exists(ledger_path)


def test_the_cli_help_says_a_malformed_tested_date_is_refused_at_the_door_with_the_setting(capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "100000")
    with pytest.raises(SystemExit):
        driver.main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "--tested-date" in text and "refused at the door" in text
    assert "nothing written" not in text.lower()


# -------- history end: the clause LEADS the driver-supplied note
#
# The gold and commodity drivers lead their note with the shared history-end clause
# (a reader of a long ledger note must not have to reach the end of it to learn the
# account was ended early), and so does this driver. The writer's own fixed metadata prefix stays ahead of everything
# (it is not editable from here); the note this driver supplies has no other
# mandatory leading text, so the clause is its first sentence-group.

def test_the_history_end_clause_leads_the_driver_supplied_note_on_the_scored_row_and_the_report(tmp_path):
    frames, cutoff = _scored_frames_and_cutoff()
    result, ledger_path, _ = _run_he(tmp_path, _d(cutoff), dry_run=False, frames=frames,
                                     max_history_drop=_ACCEPT_ANY_DROP)
    assert result["stage"] in ("survived", "cut_dsr", "cut_holdout", "cut_risk_refusal")
    clause = history_cutoff.history_end_clause(result["history_end"])
    report_note = result["cost_model_note"]
    assert report_note.startswith(f"{clause} cost model: ")
    assert report_note.count(clause) == 1 and not report_note.endswith(clause)
    outcomes = [r for r in trial_ledger.read_all(ledger_path) if r.get("event") == "outcome"]
    assert len(outcomes) == 1
    note = outcomes[0]["note"]
    assert note.split("; ", 3)[3] == report_note              # the row carries the driver's note verbatim after the writer's prefix
    assert note.count(history_cutoff.HISTORY_END_MARKER) == 1
    assert note.index(clause) < note.index("cost model: ")


def test_with_no_history_end_the_note_still_leads_with_the_cost_model_and_carries_no_clause(tmp_path):
    off, _, _ = _run_he(tmp_path, None)
    assert off["cost_model_note"].startswith("cost model: ")
    assert history_cutoff.HISTORY_END_MARKER not in off["cost_model_note"]


# -------- history end: a truncated final window says what to do about the setting
#
# Under a history end the data can end inside an open window because the SETTING ended
# it there, and the advice to re-run after that window's exit session closes is then
# the wrong advice (the operator can also just choose another cutoff). The reason_code,
# the exit status (0: it is not a history_end_* refusal) and the results-file behaviour
# are the same as with the setting off, where the reason keeps its ordinary text.

_UNSET_TRUNCATED_REASON = (
    "the data ends inside an open window (its scheduled exit lies after the last bar), so the final "
    "trade would be closed early at the last close instead of at the pre-registered exit; run a real "
    "score after the close of that window's exit session T")


def test_the_truncated_final_window_reason_with_the_setting_off_is_the_ordinary_text(tmp_path):
    frames, _, _ = _frames_cut_after_entry(1)
    real, ledger_path = _run(tmp_path, dry_run=False, synthetic=False, frames=frames)
    assert real["reason_code"] == "truncated_final_window" and real["reason"] == _UNSET_TRUNCATED_REASON
    assert not os.path.exists(ledger_path)


def test_a_truncated_final_window_under_a_history_end_names_the_cutoff_and_the_setting(tmp_path):
    frames, leg_index, entry, exit_ = _window_exit()
    inside = leg_index[leg_index.get_loc(entry) + 1]         # T-2: the setting ended the history inside the window
    assert entry < inside < exit_ and frames["IEF"].index[-1] > exit_
    real, ledger_path, _ = _run_he(tmp_path, _d(inside), dry_run=False, frames=frames,
                                   max_history_drop=_ACCEPT_ANY_DROP)
    assert real["reason_code"] == "truncated_final_window" and "verdict" not in real
    reason = " ".join(real["reason"].split())
    assert _d(inside) in reason and "--history-end" in reason
    assert "ended the scored history inside" in reason and "later dates were dropped" in reason
    assert "scheduled window exit session" in reason and "or wait" in reason
    assert reason != _UNSET_TRUNCATED_REASON and "run a real score after the close" not in reason
    assert not os.path.exists(ledger_path)
    json.dumps(real, allow_nan=False)


def test_a_truncated_final_window_where_the_setting_dropped_nothing_says_so(tmp_path):
    """The vendor's own last bar is the cutoff and it sits inside a window: the setting
    dropped nothing, and the reason must not claim it did."""
    frames, _, _ = _frames_cut_after_entry(1)
    cutoff = _d(frames["IEF"].index[-1])
    real, _, _ = _run_he(tmp_path, cutoff, dry_run=False, frames=frames, max_history_drop=0)
    assert real["reason_code"] == "truncated_final_window"
    reason = " ".join(real["reason"].split())
    assert cutoff in reason and "--history-end" in reason and "dropped nothing" in reason
    assert "later dates were dropped" not in reason and "scheduled window exit session" in reason


def test_cli_a_truncated_final_window_under_a_history_end_still_exits_zero_and_writes_its_results_file(
        tmp_path, monkeypatch, capsys):
    frames, leg_index, entry, exit_ = _window_exit()
    inside = leg_index[leg_index.get_loc(entry) + 1]
    code, ledger_path, run_dir = _cli_real_run(
        tmp_path, monkeypatch, frames, "--history-end", _d(inside), "--max-history-drop", str(_ACCEPT_ANY_DROP))
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["output"] and printed["result"]["reason_code"] == "truncated_final_window"
    assert _d(inside) in printed["result"]["reason"]
    assert _json_files(run_dir) == ["treasury-month-end-overlay-results.json"]
    assert not os.path.exists(ledger_path)


# -------- doc truth: what the non-history-end refusals really write
#
# A data-calendar gap and a stale OHLC cache are refused with a reason code and NO ledger
# row, but this driver's main() still writes a small results JSON carrying the refusal and
# exits 0 for them (only the history_end_* codes skip the file and exit 2); the per-run OHLC
# cache stays. The docstring and the comment at the refusal say so.

def test_cli_a_calendar_gap_refusal_writes_a_small_results_file_and_exits_zero_and_no_ledger_row(
        tmp_path, monkeypatch, capsys):
    frames = _frames()
    gone = frames["IEF"].index[900]
    holed = {symbol: frame.drop(gone) for symbol, frame in frames.items()}
    code, ledger_path, run_dir = _cli_real_run(tmp_path, monkeypatch, holed)
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["output"] and printed["result"]["reason_code"] == "missing_exchange_session"
    assert _json_files(run_dir) == ["treasury-month-end-overlay-results.json"]
    with open(os.path.join(run_dir, "treasury-month-end-overlay-results.json"), "r", encoding="utf-8") as fh:
        assert json.load(fh)["result"]["reason_code"] == "missing_exchange_session"
    assert not os.path.exists(ledger_path)
    assert os.path.isdir(os.path.join(run_dir, "artifacts", "ohlc"))                 # the per-run cache is kept


def test_cli_a_stale_cache_refusal_writes_a_small_results_file_and_exits_zero_and_no_ledger_row(
        tmp_path, monkeypatch, capsys):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path, synthetic=True)
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(_frames()))
    base = ["--quotes", quotes_path, "--catalog", catalog_path, "--ledger", ledger_path,
            "--research-root", research_root, "--run-id", "cli-stale"]
    assert driver.main(base + ["--dry-run"]) == 0            # a dry run caches the history under this run id
    capsys.readouterr()
    real_quotes = str(tmp_path / "real-quotes.json")
    with open(real_quotes, "w", encoding="utf-8") as fh:
        json.dump(_quotes_payload(synthetic=False), fh)
    real_base = list(base)
    real_base[1] = real_quotes
    assert driver.main(real_base + ["--tested-date", "2026-09-23"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["output"] and printed["result"]["reason_code"] == "stale_ohlc_cache"
    assert sorted(_json_files(os.path.join(research_root, "cli-stale"))) == [
        "treasury-month-end-overlay-dry-run.json", "treasury-month-end-overlay-results.json"]
    assert not os.path.exists(ledger_path)


def test_the_docstring_and_the_refusal_comment_state_the_small_results_file_for_the_other_refusals():
    doc = " ".join(driver.__doc__.split())
    said = "main() still writes a small results JSON carrying the refusal and exits 0"
    assert doc.count(said) == 2                              # the stale-cache refusal and the calendar-gap list
    assert "validate_guard_arguments" in doc                 # the door also checks the guard's two inputs
    with open(driver.__file__, "r", encoding="utf-8") as fh:
        source = " ".join(fh.read().replace("#", " ").split())     # comment markers do not split a sentence
    assert "no ledger row is written; main() still writes a small results JSON" in source
