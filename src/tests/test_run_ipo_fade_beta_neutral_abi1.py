"""Tests for scripts/run_ipo_fade_beta_neutral_abi1.py -- the bespoke ABI-1
driver for ipo_long_run_underperformance_fade_beta_neutral.

Every run below uses SYNTHETIC price frames injected through the driver's
``ohlc_loader`` hook and a TEMPORARY ledger; nothing here reads the network or
touches data/mining/trial_ledger.jsonl. The dry-run tests prove wiring only --
never a real cost measurement (the quotes file is synthetic on purpose).
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

import run_ipo_fade_beta_neutral_abi1 as driver  # noqa: E402
from mining import cost_tax_inputs, trial_ledger  # noqa: E402

_FAMILY = "ipo_long_run_underperformance_fade_beta_neutral"
_CATALOG_PATH = os.path.join(REPO_ROOT, "data", "mining", "catalog.json")

# Deliberately DIFFERENT spreads / seeds for IPO vs SPY everywhere below: with
# identical fixtures the suite would have zero power to catch an IPO-cost /
# SPY-cost or IPO-frame / SPY-frame mixup.
_IPO_QUOTE = {"bid": 50.00, "ask": 50.06}      # ~12 bps
_SPY_QUOTE = {"bid": 750.00, "ask": 750.15}    # ~2 bps


def _ohlc(close, index):
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.001,
                         "Low": np.minimum(open_, close) * 0.999,
                         "Close": close}, index=index)


def _frames(n_ipo=1900, lead=300, k=1.3, noise=0.006, ipo_drift=-0.0002, seed=5):
    spy_idx = pd.bdate_range("2010-01-04", periods=n_ipo + lead)
    rng = np.random.RandomState(seed)
    r_spy = rng.normal(0.0004, 0.01, len(spy_idx))
    r_spy[0] = 0.0
    spy = _ohlc(100.0 * np.cumprod(1.0 + r_spy), spy_idx)
    ipo_idx = spy_idx[lead:]
    rng2 = np.random.RandomState(seed + 100)
    r_ipo = k * r_spy[lead:] + ipo_drift + rng2.normal(0.0, noise, len(ipo_idx))
    r_ipo[0] = 0.0
    ipo = _ohlc(50.0 * np.cumprod(1.0 + r_ipo), ipo_idx)
    return ipo, spy


def _loader(ipo, spy):
    frames = {"IPO": ipo, "SPY": spy}

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
        "quotes": {
            "IPO": {**_IPO_QUOTE, "security_status": "Normal", "suspect": False},
            "SPY": {**_SPY_QUOTE, "security_status": "Normal", "suspect": False},
        },
    }


def _fixture(tmp_path, *, synthetic=True, status="eligible", parameters=None):
    with open(_CATALOG_PATH, "r", encoding="utf-8") as fh:
        entry = next(e for e in json.load(fh) if e["family_id"] == _FAMILY)
    entry = dict(entry)
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


def _run(tmp_path, *, dry_run=True, synthetic=True, frames=None, **fixture_kwargs):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(
        tmp_path, synthetic=synthetic, **fixture_kwargs)
    ipo, spy = frames or _frames()
    result = driver.compute_and_record(
        catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path,
        run_id="ipo-bn-test", tested_date="2026-09-23", dry_run=dry_run,
        signal_source_path="scripts/run_ipo_fade_beta_neutral_abi1.py",
        signal_source_sha256="deadbeef", research_root=research_root,
        ohlc_loader=_loader(ipo, spy))
    return result, ledger_path


# ------------------------------------------------------------ dry-run wiring

def test_dry_run_evaluates_end_to_end_without_writing_a_ledger_row(tmp_path):
    result, ledger_path = _run(tmp_path)
    assert result["family_id"] == _FAMILY
    assert "reason_code" not in result        # no skip, no compute_error
    assert result["verdict"] in ("not_scoreable", "cut_dsr", "cut_risk_refusal",
                                 "holdout_not_evaluable", "cut_holdout", "survived")
    assert result["tax_policy"] == "realization"
    assert result["benchmark"] == "SPY"
    assert result["tickers"] == ["IPO"]
    assert not os.path.exists(ledger_path)    # dry run never writes


def test_dry_run_reports_the_required_structural_details(tmp_path):
    result, _ = _run(tmp_path)
    diag = result["beta_neutral_diagnostics"]
    assert diag["n_segments"] == result["n_trades"] > 50
    assert -1.0 < diag["worst_segment_net_ret"] < diag["best_segment_net_ret"]
    assert diag["segments_at_or_below_minus_100pct"] == 0
    assert diag["hedge_ratio_min"] <= diag["hedge_ratio_median"] <= diag["hedge_ratio_max"]
    assert diag["warmup_sessions"] >= 253
    assert diag["first_overlay_date"] > "2011"          # inside the IPO span
    assert result["time_in_market"] == 1.0              # the SPY core is always on
    json.dumps(result, allow_nan=False, default=str)    # nothing non-finite leaks


def test_dry_run_reports_scored_excess_beta_next_to_the_naked_short_confound(tmp_path):
    result, _ = _run(tmp_path)
    diag = result["beta_neutral_diagnostics"]
    x = diag["overlay_notional_fraction"]
    assert x == 0.25
    assert abs(diag["scored_excess_beta_to_spy"]) / x < diag["residual_beta_tolerance_per_notional"]
    assert diag["beta_neutral_within_tolerance"] is True
    assert diag["naked_short_excess_beta_to_spy"] < -1.5     # the confound removed
    assert abs(diag["scored_excess_beta_to_spy"]) < 0.05 * abs(
        diag["naked_short_excess_beta_to_spy"])


def test_dry_run_reports_the_tax_identity_approximation_it_relies_on(tmp_path):
    result, _ = _run(tmp_path)
    diag = result["beta_neutral_diagnostics"]
    assert 0.5 < diag["tax_identity_ratio_min"] <= 1.0 <= diag["tax_identity_ratio_max"] < 2.0


def test_dry_run_reports_a_heuristic_sensitivity_of_the_tax_identity_error(tmp_path):
    """netted_tax_factors' entry-equity identity is only approximately valid
    for this account (the SPY core moves too) and the error is NOT uniformly
    conservative, so the report carries a measured HEURISTIC SENSITIVITY (not
    a rigorous bound -- the carryforward makes the true effect nonlinear): the
    tax's whole effect on the scored daily-excess Sharpe (taxed minus untaxed
    evaluation) times the worst deviation of the dollar-attribution ratio
    from 1."""
    result, _ = _run(tmp_path)
    diag = result["beta_neutral_diagnostics"]
    taxed, untaxed = diag["sharpe_daily_excess_taxed"], diag["sharpe_daily_excess_untaxed"]
    assert taxed != untaxed                        # tax genuinely bites here
    assert taxed == pytest.approx(result["sharpe_daily_excess"])
    effect = diag["tax_effect_on_daily_excess_sharpe"]
    assert effect == pytest.approx(taxed - untaxed)
    worst = max(abs(diag["tax_identity_ratio_min"] - 1.0),
                abs(diag["tax_identity_ratio_max"] - 1.0))
    assert diag["tax_identity_sensitivity_on_sharpe"] == pytest.approx(worst * abs(effect))
    assert "tax_identity_effect_bound_on_sharpe" not in diag   # no overclaiming key


def test_costs_come_from_the_right_symbols_quotes(tmp_path):
    result, _ = _run(tmp_path)
    ipo_bps = cost_tax_inputs.compute_spread_bps(_IPO_QUOTE["bid"], _IPO_QUOTE["ask"])
    spy_bps = cost_tax_inputs.compute_spread_bps(_SPY_QUOTE["bid"], _SPY_QUOTE["ask"])
    diag = result["beta_neutral_diagnostics"]
    assert diag["ipo_cost_bps"] == pytest.approx(ipo_bps)
    assert diag["spy_cost_bps"] == pytest.approx(spy_bps)
    assert diag["benchmark_entry_cost_bps"] == pytest.approx(spy_bps / 2.0)


def test_the_ipo_frame_really_drives_the_hedge_ratio(tmp_path):
    ipo, spy = _frames()
    a, _ = _run(_mk(tmp_path, "a"), frames=(ipo, spy))
    idx = ipo.index
    rng = np.random.RandomState(1)     # a low-beta IPO (0.3): hedge ratio near 0.3
    r_spy = spy["Close"].pct_change().loc[idx].to_numpy()
    r_low = 0.3 * r_spy + rng.normal(0.0, 0.006, len(idx))
    r_low[0] = 0.0
    other_ipo = _ohlc(50.0 * np.cumprod(1.0 + r_low), idx)
    b, _ = _run(_mk(tmp_path, "b"), frames=(other_ipo, spy))
    assert a["beta_neutral_diagnostics"]["hedge_ratio_median"] > 1.0
    assert 0.1 < b["beta_neutral_diagnostics"]["hedge_ratio_median"] < 0.6


def _mk(tmp_path, name):
    path = tmp_path / name
    path.mkdir()
    return path


# ---------------------------------------------------------------- guards

def test_real_run_refuses_a_synthetic_quotes_file(tmp_path):
    with pytest.raises(ValueError, match="SYNTHETIC"):
        _run(tmp_path, dry_run=False, synthetic=True)


def test_missing_catalog_entry_and_ineligible_status_skip_cleanly(tmp_path):
    result, _ = _run(_mk(tmp_path, "inelig"), status="ruled_out")
    assert result["reason_code"] == "catalog_status"
    catalog_path, quotes_path, ledger_path, research_root = _fixture(_mk(tmp_path, "none"))
    with open(catalog_path, "w", encoding="utf-8") as fh:
        json.dump([], fh)
    ipo, spy = _frames()
    skipped = driver.compute_and_record(
        catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path,
        run_id="x", tested_date="2026-09-23", dry_run=True,
        signal_source_path="p", signal_source_sha256="s", research_root=research_root,
        ohlc_loader=_loader(ipo, spy))
    assert skipped["reason_code"] == "not_in_catalog"


def test_parameter_drift_blocks_the_run(tmp_path):
    drifted = {"position": "beta_neutral_short_overlay_on_benchmark", "hedge_leg": "SPY",
               "beta_window_sessions": 126, "rebalance": "monthly_first_session",
               "overlay_notional_fraction": 0.25, "benchmark": "SPY"}
    result, _ = _run(tmp_path, parameters=drifted)
    assert result["reason_code"] == "parameter_drift"


def test_an_existing_abi1_outcome_blocks_a_second_trial(tmp_path):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path)
    trial_ledger.record_outcome(ledger_path, _FAMILY, "earlier", "2026-09-01",
                                stage="cut_dsr", dsr_value=0.1, sharpe_in_sample=0.0,
                                construction="ABI-1", note="prior")
    ipo, spy = _frames()
    result = driver.compute_and_record(
        catalog_path=catalog_path, quotes_path=quotes_path, ledger_path=ledger_path,
        run_id="again", tested_date="2026-09-23", dry_run=True,
        signal_source_path="p", signal_source_sha256="s", research_root=research_root,
        ohlc_loader=_loader(ipo, spy))
    assert result["reason_code"] == "already_scored_abi1"


# ---------------------------------------------------- ledger path (tmp ledger)

def test_a_non_dry_run_against_a_temp_ledger_writes_exactly_one_abi1_row_with_disclosures(tmp_path):
    result, ledger_path = _run(tmp_path, dry_run=False, synthetic=False)
    rows = trial_ledger.read_all(ledger_path)
    outcomes = [r for r in rows if r.get("event") == "outcome"]
    assert len(outcomes) == 1
    # abi1_driver.record_abi1_outcome writes the outcome and then registers the
    # family's first_test window: exactly these two events, nothing more
    assert sorted(r.get("event") for r in rows) == ["first_test", "outcome"]
    row = outcomes[0]
    assert row["family_id"] == _FAMILY and row["construction"] == "ABI-1"
    # the fixture scores (fixed seed), so the disclosure note IS written --
    # asserted, not skipped, so this check can never pass vacuously
    assert result["stage"] in ("survived", "cut_dsr", "cut_holdout", "cut_risk_refusal")
    assert row["stage"] == result["stage"]
    note = row.get("note") or ""
    for phrase in ("NOT MODELLED", "borrow", "financing", "beta-neutral",
                   "different strategy definition",
                   "style exposure", "heuristic sensitivity",
                   # a T-bill series exists in the pipeline and is a LOWER
                   # BOUND on financing -- the note must not claim there is no
                   # source at all -- and short-rebate income is a
                   # CONSERVATIVE omission, not a flattering one
                   "lower bound", "omitting it is conservative",
                   "price data: IPO fetched through", "SPY fetched through"):
        assert phrase.lower() in note.lower(), phrase
    assert "no source for any" not in note.lower()      # must not over-claim
    # never the real ledger
    assert os.path.abspath(ledger_path).startswith(os.path.abspath(str(tmp_path)))


def test_dry_run_reports_the_risk_layer_on_the_overlay_only_excess_with_a_whole_account_reference(tmp_path):
    """The risk-basis rule: this family's curve is a
    benchmark-plus-overlay account too, so its risk layer is measured on the
    overlay-only excess and reported next to the whole-account numbers."""
    result, _ = _run(tmp_path)
    risk = result["beta_neutral_diagnostics"]["risk_layer"]
    assert risk["account_basis"] == "overlay_only_excess"
    assert risk["whole_account_reference"]["basis"] == "whole_account"
    assert risk["whole_account_reference"]["gates_verdict"] is False
    assert isinstance(risk["failed_refusals"], list)
    json.dumps(result, allow_nan=False, default=str)


def test_a_curve_that_lost_its_account_kind_is_refused_before_anything_is_recorded(tmp_path, monkeypatch):
    real = driver.mod.build_curve

    def stripped(*args, **kwargs):
        curve = dict(real(*args, **kwargs))
        curve.pop("account_kind")
        return curve
    monkeypatch.setattr(driver.mod, "build_curve", stripped)
    with pytest.raises(RuntimeError, match="overlay_only_excess"):
        _run(tmp_path, dry_run=False, synthetic=False)
    assert not os.path.exists(str(tmp_path / "ledger.jsonl"))


def test_the_ledger_note_states_the_risk_basis(tmp_path):
    """The risk-basis rule asks for the risk basis in EVERY overlay family's
    ledger note, alongside the whole-account reference numbers."""
    result, _ = _run(tmp_path)
    note = result["cost_model_note"].lower()
    for phrase in ("risk layer (the risk-basis rule", "risk_account_basis=overlay_only_excess",
                   "overlay-only excess account", "in units of whole-account equity",
                   "whole-account reference", "gating nothing", "refusals failed in this run"):
        assert phrase in note, phrase
    risk = result["beta_neutral_diagnostics"]["risk_layer"]
    assert f"max drawdown {round(risk['account_max_drawdown'], 4)}" in result["cost_model_note"]


def test_a_core_that_differs_from_the_benchmark_before_the_first_overlay_is_refused(tmp_path, monkeypatch):
    """The excess-is-overlay-only guard runs here too, not only in the
    treasury driver. Here the overlay is off until the first monthly rebalance after the
    252-session warm-up, and the account must equal the benchmark until then."""
    real = driver.mod.build_curve

    def drifted(*args, **kwargs):
        curve = dict(real(*args, **kwargs))
        daily = curve["strategy_daily"].copy()
        first = curve["trades_df"]["entry_date"].iloc[0]
        daily.iloc[daily.index.get_loc(first) - 20] += 1e-4          # a real return during the warm-up
        curve["strategy_daily"] = daily
        return curve
    monkeypatch.setattr(driver.mod, "build_curve", drifted)
    with pytest.raises(RuntimeError, match="not overlay-only"):
        _run(tmp_path, dry_run=False, synthetic=False)
    assert not os.path.exists(str(tmp_path / "ledger.jsonl"))


def test_the_warmup_mask_marks_every_session_before_the_first_overlay_entry_off():
    built = _built()
    curve = built[2]
    mask = driver.overlay_mask(curve)
    first = curve["trades_df"]["entry_date"].iloc[0]
    assert mask.index.equals(curve["strategy_daily"].index)
    assert not mask[mask.index < first].any() and mask[mask.index >= first].all()
    assert 0 < (~mask).sum() == curve["strategy_daily"].index.get_loc(first)
    empty = dict(curve, trades_df=curve["trades_df"].iloc[0:0])
    assert not driver.overlay_mask(empty).any()             # no overlay ever: the whole account must equal SPY


def test_the_risk_layer_report_reads_the_taxed_scored_result_not_the_untaxed_counterfactual():
    """Discriminating test: a driver that read the untaxed result instead of
    the taxed, scored one must fail here."""
    import copy
    built = _built()
    taxed = copy.deepcopy(built[3])
    taxed["layer2"]["red_week_streak"].update(account=23, benchmark=19)     # values no real run produces
    assert built[4]["layer2"]["red_week_streak"]["account"] != 23           # the untaxed result differs
    risk = _diagnostics(built, result=taxed)["risk_layer"]
    assert (risk["red_week_streak_account"], risk["red_week_streak_benchmark"]) == (23, 19)


def test_the_driver_default_ledger_is_never_used_by_these_tests():
    # guard against a future test accidentally omitting ledger_path
    real = os.path.abspath(driver.LEDGER_PATH)
    assert real.endswith(os.path.join("data", "mining", "trial_ledger.jsonl"))


# ------------------------------------------------------------------- cli

def test_cli_dry_run_writes_a_json_result_and_exits_zero(tmp_path, capsys, monkeypatch):
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path)
    ipo, spy = _frames()
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(ipo, spy))
    code = driver.main(["--quotes", quotes_path, "--dry-run", "--catalog", catalog_path,
                        "--ledger", ledger_path, "--research-root", research_root,
                        "--run-id", "cli-test"])
    assert code == 0
    output = os.path.join(research_root, "cli-test", "ipo-fade-beta-neutral-dry-run.json")
    with open(output, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["dry_run"] is True and payload["construction"] == "ABI-1"
    assert payload["result"]["family_id"] == _FAMILY
    assert not os.path.exists(ledger_path)


def test_a_real_run_requires_an_explicit_tested_date(tmp_path, monkeypatch):
    """A permanent ledger row must never inherit a stale hardcoded default
    date: --tested-date is mandatory unless --dry-run."""
    catalog_path, quotes_path, ledger_path, research_root = _fixture(tmp_path, synthetic=False)
    ipo, spy = _frames()
    monkeypatch.setattr(driver.batch5, "default_ohlc_loader", _loader(ipo, spy))
    with pytest.raises(SystemExit):
        driver.main(["--quotes", quotes_path, "--catalog", catalog_path,
                     "--ledger", ledger_path, "--research-root", research_root,
                     "--run-id", "no-date"])
    assert not os.path.exists(ledger_path)
    assert not os.path.exists(os.path.join(research_root, "no-date"))


def test_result_reports_the_provenance_of_both_price_histories(tmp_path):
    """A real run under a re-used run_id silently re-uses OHLC cached by an
    earlier dry run; the result must say where each history came from and
    where each ends, so a stale cache is visible in the permanent record."""
    # SPY history runs 5 sessions past the IPO's last bar, so the two
    # last-bar dates differ and a swapped field would be visible
    long_ipo, long_spy = _frames(n_ipo=1905)
    ipo, spy = long_ipo.iloc[:1900], long_spy
    assert spy.index[-1] > ipo.index[-1]
    first, _ = _run(_mk(tmp_path, "one"), frames=(ipo, spy))
    assert first["ipo_ohlc_source"] == "fetched" and first["spy_ohlc_source"] == "fetched"
    assert first["ipo_last_bar_date"] == ipo.index[-1].strftime("%Y-%m-%d")
    assert first["spy_last_bar_date"] == spy.index[-1].strftime("%Y-%m-%d")
    assert first["ipo_last_bar_date"] != first["spy_last_bar_date"]
    # second run over the SAME research root re-uses the caches; deleting only
    # the IPO cache makes exactly one history "fetched" and the other "cache"
    catalog_path, quotes_path, ledger_path, research_root = _fixture(_mk(tmp_path, "two"))
    common = dict(catalog_path=catalog_path, quotes_path=quotes_path,
                  ledger_path=ledger_path, run_id="ipo-bn-test", tested_date="2026-09-23",
                  dry_run=True, signal_source_path="p", signal_source_sha256="s",
                  research_root=research_root)
    driver.compute_and_record(**common, ohlc_loader=_loader(ipo, spy))
    again = driver.compute_and_record(**common, ohlc_loader=_loader(ipo, spy))
    assert again["ipo_ohlc_source"] == "cache" and again["spy_ohlc_source"] == "cache"
    os.remove(os.path.join(research_root, "ipo-bn-test", "artifacts", "ohlc", "IPO.csv"))
    mixed = driver.compute_and_record(**common, ohlc_loader=_loader(ipo, spy))
    assert mixed["ipo_ohlc_source"] == "fetched" and mixed["spy_ohlc_source"] == "cache"


# --------------------------------------------- checks vs independent maths

def _built(tmp_path_unused=None):
    from mining import abi1_driver, benchmark_relative
    ipo, spy = _frames()
    ipo_cost = cost_tax_inputs.compute_spread_bps(_IPO_QUOTE["bid"], _IPO_QUOTE["ask"])
    spy_spread = cost_tax_inputs.compute_spread_bps(_SPY_QUOTE["bid"], _SPY_QUOTE["ask"])
    spy_cost, entry_cost = spy_spread, spy_spread / 2.0
    curve = driver.mod.build_curve(ipo, spy, ipo_cost_bps=ipo_cost, spy_cost_bps=spy_cost,
                                   spy_entry_cost_bps=entry_cost)
    kwargs = dict(benchmark_frame=spy, benchmark_entry_cost_bps=entry_cost,
                  account_type="taxable")
    taxed = abi1_driver.evaluate_family(curve, tax_policy="realization", **kwargs)
    untaxed = abi1_driver.evaluate_family(curve, tax_policy="none", **kwargs)
    bench = benchmark_relative.cost_adjusted_benchmark_daily_returns(
        spy, entry_cost_bps=entry_cost)
    return ipo, spy, curve, taxed, untaxed, bench, ipo_cost, spy_cost, entry_cost


def _diagnostics(built, *, result=None):
    ipo, spy, curve, taxed, untaxed, bench, ipo_cost, spy_cost, entry_cost = built
    return driver.beta_neutral_diagnostics(
        curve, taxed if result is None else result, untaxed_result=untaxed,
        ipo_frame=ipo, benchmark_daily=bench, ipo_cost_bps=ipo_cost,
        spy_cost_bps=spy_cost, benchmark_entry_cost_bps=entry_cost)


def test_details_equal_independent_aggregates_of_the_trades_frame():
    built = _built()
    ipo, spy, curve = built[0], built[1], built[2]
    diag = _diagnostics(built)
    trades = curve["trades_df"]
    assert diag["hedge_ratio_min"] == trades["hedge_ratio"].min()
    assert diag["hedge_ratio_max"] == trades["hedge_ratio"].max()
    assert diag["hedge_ratio_median"] == pytest.approx(trades["hedge_ratio"].median())
    assert diag["worst_segment_net_ret"] == trades["net_ret"].min()
    assert diag["best_segment_net_ret"] == trades["net_ret"].max()
    assert diag["segments_at_or_below_minus_100pct"] == int((trades["net_ret"] <= -1.0).sum())
    from mining import beta_neutral_overlay as overlay
    first = int(next(p for p in overlay.rebalance_positions(ipo.index) if p >= 253))
    assert diag["warmup_sessions"] == first
    assert diag["first_overlay_date"] == ipo.index[first].strftime("%Y-%m-%d")
    assert diag["overlay_sessions"] == len(ipo) - first


def test_tax_identity_ratio_is_growth_from_the_prior_close_to_the_exit_close():
    built = _built()
    ipo, curve = built[0], built[2]
    diag = _diagnostics(built)
    account = curve["strategy_daily"].to_numpy()
    trades = curve["trades_df"]
    ratios = []
    for _, row in trades.iterrows():
        entry = ipo.index.get_loc(row["entry_date"])
        exit_ = ipo.index.get_loc(row["exit_date"])
        growth = np.prod(1.0 + account[entry:exit_ + 1])     # prior close -> exit close
        ratios.append(growth / (1.0 + row["net_ret"]))
    assert diag["tax_identity_ratio_min"] == pytest.approx(min(ratios), rel=1e-12)
    assert diag["tax_identity_ratio_max"] == pytest.approx(max(ratios), rel=1e-12)


def test_scored_beta_is_read_from_the_taxed_result_and_naked_short_carries_its_cost():
    from mining import benchmark_relative
    from mining import beta_neutral_overlay as overlay
    built = _built()
    ipo, taxed, untaxed, bench, ipo_cost = built[0], built[3], built[4], built[5], built[6]
    diag = _diagnostics(built)
    from_taxed = benchmark_relative.ols_alpha_beta(taxed["log_excess"], bench)["beta"]
    from_untaxed = benchmark_relative.ols_alpha_beta(untaxed["log_excess"], bench)["beta"]
    assert from_taxed != from_untaxed                 # tax genuinely bites in the fixture
    assert diag["scored_excess_beta_to_spy"] == pytest.approx(from_taxed, rel=1e-12)
    naked = overlay.scored_excess_spy_beta(
        overlay.naked_short_account_daily(ipo, cost_bps=ipo_cost), bench)["beta"]
    assert diag["naked_short_excess_beta_to_spy"] == pytest.approx(naked, rel=1e-12, abs=0)
    zero_cost = overlay.scored_excess_spy_beta(
        overlay.naked_short_account_daily(ipo, cost_bps=0.0), bench)["beta"]
    assert naked != zero_cost                         # the cost is genuinely in the figure


def test_the_neutrality_flag_is_per_unit_of_notional_not_the_raw_beta():
    """A raw beta of -0.15 looks tiny next to a 0.25 tolerance, but at a 0.25
    notional it is -0.6 per unit of notional: NOT neutral. The flag must be
    False there and True for a genuinely neutral series."""
    built = _built()
    taxed, bench = built[3], built[5]
    x = driver.mod.PARAMETERS["overlay_notional_fraction"]
    idx = taxed["log_excess"].index
    not_neutral = {**taxed, "log_excess": -0.6 * x * bench.reindex(idx)}
    neutral = {**taxed, "log_excess": 0.001 * x * bench.reindex(idx)}
    bad = _diagnostics(built, result=not_neutral)
    good = _diagnostics(built, result=neutral)
    assert bad["scored_excess_beta_per_notional"] == pytest.approx(-0.6, rel=1e-9)
    assert bad["beta_neutral_within_tolerance"] is False
    assert good["beta_neutral_within_tolerance"] is True
