import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for path in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_batch3_mining as batch3
import trading_calendar
from mining import constants, trial_ledger
from mining.backtest_engine import SignalResult


CATALOG = os.path.join(_ROOT, "data", "mining", "catalog.json")


def _frame(start="2024-01-02", n=520, offset=0.0):
    index = pd.date_range(start, periods=n, freq="B")
    values = 100.0 + offset + np.arange(n, dtype=float) * 0.05
    return pd.DataFrame({
        "Open": values, "High": values + 1.0, "Low": values - 1.0,
        "Close": values + np.sin(np.arange(n) / 7.0),
    }, index=index)


def _frame_for_index(index, offset=0.0):
    index = pd.DatetimeIndex(index)
    values = 100.0 + offset + np.arange(len(index), dtype=float) * 0.05
    return pd.DataFrame({
        "Open": values, "High": values + 1.0, "Low": values - 1.0,
        "Close": values + np.sin(np.arange(len(index)) / 7.0),
    }, index=index)


def _always_long_factory(_index):
    def signal(_frame, _i):
        return SignalResult(entry=True, exit=False)
    return signal


def _computed(family_id="family-a", trade_count=20, sharpe=0.5,
              returns=None, ticker="SPY", tickers=None,
              raw_catalog_parameters=None):
    if returns is None:
        returns = pd.Series(
            [0.0, 0.01, 0.0, -0.002, 0.004],
            index=pd.date_range("2024-01-02", periods=5, freq="B"))
    return {
        "family_id": family_id, "ticker": ticker,
        # `tickers` is the REAL
        # instrument list, independent of `ticker` -- which some producers
        # (run_swing_mining.py/run_day_trading_mining.py's own
        # `_ticker_label`) can set to an arbitrary DISPLAY LABEL like
        # "S&P 500 pool" rather than a parseable ticker. evidence's
        # instrument_set must come from here, never from splitting `ticker`.
        "tickers": tickers if tickers is not None else [ticker],
        "holdout_start": "2024-07-01", "holdout_end": "2024-12-31",
        "holdout_complete": True,
        "stats_in_sample": {"trade_count": trade_count,
                            "net_edge_bps": 100.0, "sharpe": sharpe,
                            "skew": 0.0, "kurtosis": 3.0},
        "stats_holdout": {"trade_count": 2, "net_edge_bps": 25.0,
                          "sharpe": 0.1, "skew": 0.0, "kurtosis": 3.0},
        "holdout_net_edge_bps": 25.0, "daily_returns": returns,
        # in_sample_start/in_sample_end feed evidence.build_insufficient_
        # trades when trade_count is below MIN_TRADES_FOR_SCORING -- both
        # strictly before holdout_start,
        # matching compute_candidate's own real windows["in_sample_index"].
        "in_sample_start": "2020-01-02", "in_sample_end": "2024-06-28",
        "catalog_parameters": {},
        # `raw_catalog_parameters` is
        # the UNMAPPED catalog dict, independent of `catalog_parameters`
        # (`catalog_parameters()`'s own factory-ADAPTER view, which renames/
        # drops/casts keys for some families). evidence's observed.
        # parameters must come from here.
        "raw_catalog_parameters": (raw_catalog_parameters
                                   if raw_catalog_parameters is not None else {}),
        "note": "test computation",
    }


def _valid_finra_metadata():
    return {
        "source_url": batch3.external_data.FINRA_MARGIN_STATISTICS_URL,
        "retrieved_at": "2026-01-15T12:00:00.000000Z",
        "raw_sha256": "a" * 64,
        "worksheet": batch3.external_data.FINRA_WORKSHEET,
        "units": "USD millions",
        "date_semantics": "reference_month_end",
    }


def test_holdout_window_is_derived_only_from_preregistered_constant():
    window = batch3.derive_holdout_window("2026-08-29")
    assert constants.HOLDOUT_WINDOW_MONTHS == 6
    assert window == {"holdout_start": "2026-02-28",
                      "holdout_end": "2026-08-29",
                      "required_session": "2026-08-28"}


def test_required_session_uses_vendored_nyse_calendar_on_weekday_holiday():
    window = batch3.derive_holdout_window("2026-07-03")
    assert window["required_session"] == "2026-07-02"


@pytest.mark.parametrize("tested_date", ["2025-12-31", "2029-01-01"])
def test_holdout_window_fails_closed_outside_calendar_coverage(tested_date):
    with pytest.raises(ValueError, match="calendar coverage"):
        batch3.derive_holdout_window(tested_date)


def test_single_asset_windows_are_bounded_before_factory_and_engine(monkeypatch):
    frame = _frame(n=20)
    start, end = frame.index[12], frame.index[-2]
    factory_seen, engine_seen = [], []

    def factory(index):
        factory_seen.append(index.copy())
        return _always_long_factory(index)
    real = batch3.backtest_engine.run_backtest

    def spy(bucket, signal, **kwargs):
        engine_seen.append(bucket.index.copy())
        return real(bucket, signal, **kwargs)
    monkeypatch.setattr(batch3.backtest_engine, "run_backtest", spy)
    result = batch3.run_single_asset_windows(
        frame, factory, holdout_start=str(start.date()),
        holdout_end=str(end.date()), cost_bps=0.0, tax_haircut_rate=0.0)
    assert factory_seen[0].max() < start
    assert factory_seen[1].min() >= start
    assert factory_seen[1].max() <= end
    assert engine_seen[0].equals(factory_seen[0])
    assert engine_seen[1].equals(factory_seen[1])
    assert result["in_sample_trades"]["exit_date"].max() < start
    assert result["holdout_trades"]["entry_date"].min() >= start
    assert result["holdout_trades"]["exit_date"].max() <= end


def test_rotation_and_signal_frames_are_bucket_bounded_and_naturally_close(
        monkeypatch):
    risk = _frame(n=30)
    defensive = _frame(n=30, offset=10.0)
    signal = _frame(n=30, offset=20.0).drop(risk.index[[3, 17]])
    start, end = risk.index[20], risk.index[-2]
    calls = []
    real = batch3.rotation_engine.run_two_asset_rotation

    def spy(asset_frames, target_fn, **kwargs):
        calls.append((
            {k: v.index.copy() for k, v in asset_frames.items()},
            {k: v.index.copy() for k, v in kwargs["signal_frames"].items()}))
        return real(asset_frames, target_fn, **kwargs)
    monkeypatch.setattr(batch3.rotation_engine, "run_two_asset_rotation", spy)
    result = batch3.run_rotation_windows(
        {"RISK": risk, "DEF": defensive}, {"SIG": signal},
        lambda frames: "RISK", holdout_start=str(start.date()),
        holdout_end=str(end.date()),
        cost_bps_by_asset={"RISK": 1.0, "DEF": 2.0},
        tax_haircut_rate=0.0)
    for assets, signals_ in calls:
        assert assets["RISK"].equals(assets["DEF"])
        assert signals_["SIG"].equals(assets["RISK"])
    assert calls[0][0]["RISK"].max() < start
    # The ENGINE's holdout call is padded backward so a real lookback signal
    # has trailing history to compute from -- it genuinely receives data
    # from BEFORE holdout_start, not just [holdout_start, holdout_end].
    # (Deliberately NOT derived from batch3.HOLDOUT_LOOKBACK_BUFFER_DAYS --
    # a check that re-reads the very constant it's supposed to verify is
    # tautological and passes even if the padding were removed entirely. The
    # buffer's actual SIZE is protected by the dedicated real-lookback-need
    # tests below, not by this alignment-focused test.)
    assert calls[1][0]["RISK"].min() < start
    assert calls[1][0]["RISK"].max() <= end
    # But everything RETURNED for holdout describes the real holdout window
    # -- the padding is an internal computation detail, never a leak.
    assert result["holdout_index"].min() >= start
    assert result["holdout_index"].max() <= end
    assert result["alignment"]["in_sample"]["signal_dates_dropped"] > 0
    assert result["in_sample_trades"]["exit_date"].max() < start
    # This fixture's trivial always-"RISK" target_fn opens its one position
    # during the padding period and holds straight to the end -- exactly
    # the spanning-trade case _mark_to_market_spanning_trades exists for,
    # so holdout_trades must contain that one marked-to-market trade, not
    # be empty.
    assert len(result["holdout_trades"]) == 1
    assert result["holdout_trades"].iloc[0]["entry_date"] == result["holdout_index"][0]
    assert (pd.to_datetime(result["holdout_trades"]["entry_date"]) >= start).all()
    assert result["holdout_trades"]["exit_date"].max() <= end


def _lookback_gated_toggle_target_fn(min_days):
    """A target_fn shaped like the real signals.py factories: it needs
    ``min_days`` of trailing history on the risk leg before it will ever
    return a decision, exactly like make_monthly_absolute_momentum_target_fn
    (lookback_months), make_monthly_volatility_target_fn (median_lookback_days),
    and make_moving_average_regime_target_fn/make_ratio_regime_target_fn (slow)
    all do via their own `len(...) < lookback: return None` guards. Once past
    the lookback, it toggles every 20 decisions so a correctly padded holdout
    produces multiple distinct trades across the holdout window, not just one
    warm-up position carried in from before holdout_start."""
    def target(frames):
        risk = frames["RISK"]
        if len(risk) < min_days:
            return None
        return "RISK" if (len(risk) - min_days) % 20 < 10 else "DEF"
    return target


def test_holdout_rotation_pads_lookback_so_a_252_day_signal_can_actually_decide():
    n = 700
    risk = _frame(n=n)
    defensive = _frame(n=n, offset=10.0)
    # Holdout = last ~130 sessions (~6 months), matching this pipeline's own
    # HOLDOUT_WINDOW_MONTHS=6 convention. A 252-session lookback cannot be
    # satisfied from a bare, unpadded holdout-only slice (fewer than 130 rows
    # exist in it, ever), so a naive holdout bucket starves this signal for
    # the entire window -- the exact defect this test exists to catch.
    start, end = risk.index[n - 130], risk.index[-1]
    target_fn = _lookback_gated_toggle_target_fn(252)

    result = batch3.run_rotation_windows(
        {"RISK": risk, "DEF": defensive}, {}, target_fn,
        holdout_start=str(start.date()), holdout_end=str(end.date()),
        cost_bps_by_asset={"RISK": 1.0, "DEF": 2.0}, tax_haircut_rate=0.0)

    holdout_trades = result["holdout_trades"]
    assert len(holdout_trades) > 0, (
        "a signal needing 252 sessions of lookback produced zero holdout "
        "trades -- the holdout bucket gave it no real trailing history")
    assert (pd.to_datetime(holdout_trades["entry_date"]) >= start).all(), (
        "a trade whose entry predates holdout_start leaked through -- "
        "warm-up/padding trades must be filtered before they're counted "
        "as real out-of-sample activity")
    assert (pd.to_datetime(holdout_trades["exit_date"]) <= end).all()
    # holdout_index/stats must describe the REAL holdout window, not the
    # padded lookback window used internally to compute it.
    assert result["holdout_index"].min() >= start
    assert result["holdout_index"].max() <= end


def _lookback_gated_persistent_target_fn(min_days):
    """Needs min_days of lookback, then locks onto RISK forever and never
    switches again -- the shape of a trending absolute-momentum signal that
    stays on one side for a whole holdout window. Used to prove a position
    opened during the lookback-padding period and held straight through
    holdout with zero switches is not silently dropped."""
    def target(frames):
        risk = frames["RISK"]
        if len(risk) < min_days:
            return None
        return "RISK"
    return target


def test_holdout_rotation_marks_to_market_a_position_that_never_switches_during_holdout():
    n = 700
    risk = _frame(n=n)
    defensive = _frame(n=n, offset=10.0)
    start, end = risk.index[n - 130], risk.index[-1]
    target_fn = _lookback_gated_persistent_target_fn(252)

    result = batch3.run_rotation_windows(
        {"RISK": risk, "DEF": defensive}, {}, target_fn,
        holdout_start=str(start.date()), holdout_end=str(end.date()),
        cost_bps_by_asset={"RISK": 1.0, "DEF": 2.0}, tax_haircut_rate=0.0)

    holdout_trades = result["holdout_trades"]
    assert len(holdout_trades) > 0, (
        "a position opened before holdout_start and held straight through "
        "the entire holdout window (never switching) must still be counted "
        "as real holdout activity, not dropped for predating holdout_start")
    assert len(holdout_trades) == 1
    row = holdout_trades.iloc[0]
    assert row["asset"] == "RISK"
    first_real_session = result["holdout_index"][0]
    # Marked to market at the FIRST real holdout session's open, not the
    # position's true (pre-holdout) entry date/price.
    assert pd.Timestamp(row["entry_date"]) == first_real_session
    expected_entry_px = float(risk.loc[first_real_session, "Open"])
    assert row["entry_px"] == pytest.approx(expected_entry_px)
    assert pd.Timestamp(row["exit_date"]) <= end
    assert row["net_ret"] != 0.0, "real, non-degenerate P&L over the real holdout span"


def test_mark_to_market_treats_exit_exactly_at_start_as_pure_warmup_not_spanning():
    idx = pd.date_range("2024-01-02", periods=10, freq="B")
    start = idx[5]
    risk_frame = pd.DataFrame({
        "Open": np.arange(10, dtype=float) + 100.0,
        "High": np.arange(10, dtype=float) + 101.0,
        "Low": np.arange(10, dtype=float) + 99.0,
        "Close": np.arange(10, dtype=float) + 100.5,
    }, index=idx)
    # A "switch" trade that closed AT holdout_start's own open (rotation_
    # engine's real convention) had zero real holdout exposure -- its whole
    # true lifecycle was in the padding period. It must be dropped as pure
    # warm-up, not marked to market: marking it to market would synthesize
    # an entry at the SAME price as its exit (gross_ret == 0.0) while still
    # charging a real round-trip cost for a position that was never
    # actually held during real holdout. An inclusive `exit_ >= start`
    # spanning test would get this wrong.
    trades = pd.DataFrame([{
        "asset": "RISK", "entry_date": idx[2], "exit_date": start,
        "entry_px": 102.0, "exit_px": 105.0, "held": 3, "reason": "switch",
        "gross_ret": 0.0294, "net_ret": 0.0284,
    }])

    real, pure_warmup, marked = batch3._mark_to_market_spanning_trades(
        trades, {"RISK": risk_frame}, start, {"RISK": 1.0},
        tax_haircut_rate=0.0)

    assert pure_warmup == 1
    assert marked == 0
    assert len(real) == 0


def test_misaligned_traded_calendar_fails_closed():
    risk = _frame(n=30)
    defensive = _frame(n=30, offset=10.0).drop(risk.index[7])
    with pytest.raises(ValueError, match="identical calendar"):
        batch3.run_rotation_windows(
            {"RISK": risk, "DEF": defensive}, {}, lambda frames: "RISK",
            holdout_start=str(risk.index[20].date()),
            holdout_end=str(risk.index[-1].date()),
            cost_bps_by_asset={"RISK": 1.0, "DEF": 2.0},
            tax_haircut_rate=0.0)


def test_rotation_accepts_and_discloses_legitimate_differing_inceptions():
    risk = _frame(n=30)
    defensive = _frame(n=30, offset=10.0).iloc[1:].copy()
    result = batch3.run_rotation_windows(
        {"RISK": risk, "DEF": defensive}, {}, lambda frames: "RISK",
        holdout_start=str(risk.index[20].date()),
        holdout_end=str(risk.index[-1].date()),
        cost_bps_by_asset={"RISK": 1.0, "DEF": 2.0},
        tax_haircut_rate=0.0)

    normalization = result["alignment"]["normalization"]
    assert normalization == {
        "raw_leg_starts": {
            "RISK": risk.index[0].strftime("%Y-%m-%d"),
            "DEF": defensive.index[0].strftime("%Y-%m-%d"),
        },
        "common_inception": defensive.index[0].strftime("%Y-%m-%d"),
        "sessions_trimmed_by_leg": {"RISK": 1, "DEF": 0},
    }


def test_rotation_rejects_leading_gap_after_common_inception():
    risk = _frame(n=30)
    defensive = _frame(n=30, offset=10.0).iloc[1:].drop(risk.index[7])
    with pytest.raises(ValueError, match="identical calendar"):
        batch3.run_rotation_windows(
            {"RISK": risk, "DEF": defensive}, {}, lambda frames: "RISK",
            holdout_start=str(risk.index[20].date()),
            holdout_end=str(risk.index[-1].date()),
            cost_bps_by_asset={"RISK": 1.0, "DEF": 2.0},
            tax_haircut_rate=0.0)


def test_rotation_rejects_missing_first_expected_holdout_session():
    risk = _frame(n=30)
    first_holdout = risk.index[20]
    expected_holdout = risk.index[20:]
    risk = risk.drop(first_holdout)
    defensive = _frame(n=30, offset=10.0).drop(first_holdout)
    with pytest.raises(ValueError, match="expected holdout calendar"):
        batch3.run_rotation_windows(
            {"RISK": risk, "DEF": defensive}, {}, lambda frames: "RISK",
            holdout_start=str(first_holdout.date()),
            holdout_end=str(expected_holdout[-1].date()),
            cost_bps_by_asset={"RISK": 1.0, "DEF": 2.0},
            tax_haircut_rate=0.0,
            expected_holdout_calendar=expected_holdout)


def test_shared_correction_prefers_the_pooled_sr_variance_when_available(tmp_path):
    # Seed the ledger with enough real-scored families that
    # pooled_sr_variance stops returning None, and confirm
    # compute_shared_correction uses THAT instead of this run's own
    # (here: degenerate, single-candidate) sample.
    ledger = str(tmp_path / "ledger.jsonl")
    pooled_values = [0.2, 0.5, 0.9, 0.1, 0.6, 0.3, 0.7, 0.4, 0.8, 0.35]
    assert len(pooled_values) >= constants.MIN_POOLED_SR_VARIANCE_SAMPLES
    for i, v in enumerate(pooled_values):
        trial_ledger.record_outcome(
            ledger, f"pool_fam_{i}", "prior-run", "2026-09-01", "cut_dsr",
            dsr_value=0.1, sharpe_in_sample=v)
    items = [_computed("solo", sharpe=0.5)]   # exactly 1 candidate this run
    result = batch3.compute_shared_correction(items, ledger)
    assert result["sr_variance"] == pytest.approx(
        trial_ledger.pooled_sr_variance(ledger))
    assert result["sr_variance"] != pytest.approx(
        constants.DEFAULT_SR_VARIANCE_FLOOR)


def test_shared_correction_falls_back_to_per_run_variance_when_pool_is_thin(
        tmp_path):
    # An empty/thin ledger falls back to this run's own per-candidate
    # sharpe variance.
    ledger = str(tmp_path / "ledger.jsonl")
    items = [_computed("a", sharpe=0.4), _computed("b", sharpe=0.6)]
    result = batch3.compute_shared_correction(items, ledger)
    assert result["sr_variance"] == pytest.approx(np.var([0.4, 0.6], ddof=1))


def test_shared_correction_falls_back_to_the_floor_when_pool_and_run_both_thin(
        tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    items = [_computed("solo", sharpe=0.5)]
    result = batch3.compute_shared_correction(items, ledger)
    assert result["sr_variance"] == pytest.approx(
        constants.DEFAULT_SR_VARIANCE_FLOOR)


def test_real_score_persists_sharpe_in_sample_on_a_real_dsr_score(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    items = [_computed("a", sharpe=0.77, trade_count=20)]
    # The fixture's own default stats_holdout carries only 2
    # trades, below MIN_HOLDOUT_TRADES -- which would defer this family
    # (insufficient evidence) instead of reaching the SCORED outcome
    # this test exists to check. Raise the holdout trade count so the
    # family still clears the floor and reaches a score.
    items[0]["stats_holdout"] = {"trade_count": constants.MIN_HOLDOUT_TRADES,
                                 "net_edge_bps": 25.0, "sharpe": 0.1,
                                 "skew": 0.0, "kurtosis": 3.0}
    batch3.score_batch(items, ledger_path=ledger, run_id="run",
                       tested_date="2024-12-31", dry_run=False)
    events = trial_ledger.read_all(ledger)
    outcome = next(e for e in events if e["event"] == "outcome")
    assert outcome["sharpe_in_sample"] == pytest.approx(0.77)


def test_real_score_does_not_fabricate_sharpe_below_min_trades(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    items = [_computed("a", sharpe=0.77,
                       trade_count=constants.MIN_TRADES_FOR_SCORING - 1)]
    batch3.score_batch(items, ledger_path=ledger, run_id="run",
                       tested_date="2024-12-31", dry_run=False)
    events = trial_ledger.read_all(ledger)
    outcome = next(e for e in events if e["event"] == "outcome")
    assert outcome["dsr_value"] is None
    assert outcome["sharpe_in_sample"] is None


def test_shared_correction_calls_independent_trial_estimate_once_and_is_ordered(
        tmp_path, monkeypatch):
    calls = []
    def n_trials(arrays):
        calls.append([np.asarray(a).copy() for a in arrays])
        return 1.25
    monkeypatch.setattr(batch3.dsr, "num_independent_trials", n_trials)
    items = [
        _computed("b", sharpe=0.4),
        _computed("a", sharpe=0.6, returns=pd.Series(
            [0.0, 0.02, 0.0],
            index=pd.date_range("2024-01-03", periods=3, freq="B"))),
        _computed("constant", sharpe=0.2, returns=pd.Series(
            0.0, index=pd.date_range("2024-01-02", periods=4, freq="B"))),
    ]
    first = batch3.compute_shared_correction(items, str(tmp_path / "ledger"))
    second = batch3.compute_shared_correction(
        list(reversed(items)), str(tmp_path / "ledger"))
    assert len(calls) == 2
    assert first == second
    assert first["n_trials"] == 3.0
    assert first["sr_variance"] == pytest.approx(np.var([.4, .6, .2], ddof=1))
    assert first["nondegenerate_series_count"] == 2


def test_no_nondegenerate_return_series_uses_raw_count_without_dsr_call(
        tmp_path, monkeypatch):
    monkeypatch.setattr(batch3.dsr, "num_independent_trials",
                        lambda _: pytest.fail("correlation must not be called"))
    items = [_computed("a"), _computed("b")]
    for item in items:
        item["daily_returns"][:] = 0.0
    correction = batch3.compute_shared_correction(items, str(tmp_path / "ledger"))
    assert correction["n_this"] == 2.0
    assert correction["n_trials"] == 2.0


def _stable_score(**_kwargs):
    return 0.97


def _result_projection(results):
    return {item["family_id"]: (
        item["stage"], item["dsr_value"], item["shared_n_trials"],
        item["shared_sr_variance"]) for item in results}


def test_dry_and_temp_ledger_real_scoring_are_equivalent_and_order_invariant(
        tmp_path, monkeypatch):
    monkeypatch.setattr(batch3.dsr, "deflated_sharpe_ratio", _stable_score)
    items = [_computed("a", sharpe=.5), _computed("b", sharpe=.6)]
    dry_ledger = str(tmp_path / "dry.jsonl")
    real_ledger = str(tmp_path / "real.jsonl")
    dry = batch3.score_batch(
        items, ledger_path=dry_ledger, run_id="run", tested_date="2024-12-31",
        dry_run=True)
    real = batch3.score_batch(
        list(reversed(items)), ledger_path=real_ledger, run_id="run",
        tested_date="2024-12-31", dry_run=False)
    assert _result_projection(dry) == _result_projection(real)
    assert trial_ledger.read_all(dry_ledger) == []
    assert [r["event"] for r in trial_ledger.read_all(real_ledger)] == [
        "first_test", "first_test", "outcome", "outcome"]


def test_mid_scoring_exception_defensively_outcomes_every_new_pin(
        tmp_path, monkeypatch):
    ledger = str(tmp_path / "ledger.jsonl")
    count = {"value": 0}
    def fail_middle(**_kwargs):
        count["value"] += 1
        if count["value"] == 2:
            raise RuntimeError("middle score failed")
        return 0.5
    monkeypatch.setattr(batch3.dsr, "deflated_sharpe_ratio", fail_middle)
    items = [_computed("a"), _computed("b"), _computed("c")]
    with pytest.raises(RuntimeError, match="middle score failed"):
        batch3.score_batch(items, ledger_path=ledger, run_id="run",
                           tested_date="2024-12-31", dry_run=False)
    records = trial_ledger.read_all(ledger)
    pinned = {r["family_id"] for r in records if r["event"] == "first_test"}
    outcomes = {r["family_id"] for r in records if r["event"] == "outcome"}
    assert pinned == {"a", "b", "c"}
    assert outcomes == pinned
    # "a" scored successfully BEFORE the mid-batch failure (a real, if
    # below-bar, DSR) -- a genuine DSR cut. "b" (whose own scoring call
    # raised) and "c" (never reached) are defensively recorded as
    # not_evaluated_error, not the SAME stage string "a" carries --
    # otherwise all three would share one identical stage string despite
    # "a" being a genuine score and "b"/"c" being mechanical failures, exactly
    # the ambiguity this lifecycle exists to make impossible to write.
    assert trial_ledger.latest_outcome(ledger, "a") == "cut_dsr"
    assert trial_ledger.latest_outcome(ledger, "b") == "not_evaluated_error"
    assert trial_ledger.latest_outcome(ledger, "c") == "not_evaluated_error"
    by_family = {r["family_id"]: r for r in records if r["event"] == "outcome"}
    assert by_family["a"]["dsr_value"] == pytest.approx(0.5)
    assert by_family["a"]["evidence"] is None
    for fid in ("b", "c"):
        assert by_family[fid]["dsr_value"] is None
        assert by_family[fid]["evidence"]["reason_code"] == "engine_error"


def test_real_score_rechecks_all_families_before_first_pin(tmp_path, monkeypatch):
    ledger = str(tmp_path / "ledger.jsonl")
    items = [_computed("a"), _computed("b")]

    def concurrent_pin(_computed_items, _ledger_path):
        assert trial_ledger.record_first_test(
            ledger, "a", "2024-12-31", "2024-07-01", "2024-12-31", [],
            "concurrent-run")
        return {"n_this": 2.0, "post_batch_ledger_floor": 2,
                "n_trials": 2.0, "sr_variance": 0.01,
                "nondegenerate_series_count": 2,
                "computed_family_count": 2}

    monkeypatch.setattr(batch3, "compute_shared_correction", concurrent_pin)
    with pytest.raises(RuntimeError, match="already pinned.*a"):
        batch3.score_batch(
            items, ledger_path=ledger, run_id="our-run",
            tested_date="2024-12-31", dry_run=False)

    records = trial_ledger.read_all(ledger)
    assert [(record["event"], record["family_id"])
            for record in records] == [("first_test", "a")]


def test_insufficient_trades_is_still_pinned_and_outcomed(tmp_path):
    ledger = str(tmp_path / "ledger.jsonl")
    result = batch3.score_batch(
        [_computed(trade_count=constants.MIN_TRADES_FOR_SCORING - 1)],
        ledger_path=ledger, run_id="run", tested_date="2024-12-31",
        dry_run=False)[0]
    # A trade-count cut below MIN_TRADES_FOR_SCORING is an insufficient-evidence
    # stage, not a scored DSR cut: no
    # DSR was ever computed on this branch, so filing it as a statistical
    # rejection would be indistinguishable from a genuine one -- the exact
    # ambiguity this lifecycle exists to close.
    assert result["stage"] == "insufficient_evidence"
    assert result["dsr_value"] is None
    assert trial_ledger.latest_outcome(ledger, "family-a") == "insufficient_evidence"
    events = trial_ledger.read_all(ledger)
    outcome = next(e for e in events if e["event"] == "outcome")
    assert outcome["evidence"]["reason_code"] == "insufficient_trades"
    assert outcome["evidence"]["observed"]["trade_count"] == (
        constants.MIN_TRADES_FOR_SCORING - 1)
    # A real measured per-trade sharpe DOES live inside the evidence
    # payload's own observed block (translated from stats['sharpe'], this
    # pipeline's own key -- evidence.py's schema names it 'sharpe_in_
    # sample') even though the ledger row's own TOP-LEVEL sharpe_in_sample
    # stays None (that field means "fed a real DSR computation", which
    # never happened here).
    assert outcome["evidence"]["observed"]["sharpe_in_sample"] == pytest.approx(0.5)
    assert outcome["sharpe_in_sample"] is None


# ---------------------------------------------------------------------------
# Guarded behaviours (A)-(C), each pinned by a discriminating test below.
# ---------------------------------------------------------------------------

def test_insufficient_evidence_uses_real_tickers_not_a_display_label(tmp_path):
    # (A): `item["ticker"]` can be an arbitrary DISPLAY LABEL (e.g.
    # run_swing_mining.py's own `_ticker_label` falls back to
    # `candidate.get("ticker_label")`, a human-readable string like
    # "S&P 500 pool", not a parseable ticker). Splitting THAT on "/" and
    # feeding it to evidence's uppercase-ticker validator would raise.
    # `tickers` (the real instrument list) must be used instead.
    ledger = str(tmp_path / "ledger.jsonl")
    item = _computed(trade_count=constants.MIN_TRADES_FOR_SCORING - 1,
                     ticker="S&P 500 pool", tickers=["AAA", "BBB"])
    result = batch3.score_batch(
        [item], ledger_path=ledger, run_id="run", tested_date="2024-12-31",
        dry_run=False)[0]
    assert result["stage"] == "insufficient_evidence"
    outcome = next(e for e in trial_ledger.read_all(ledger) if e["event"] == "outcome")
    assert outcome["evidence"]["observed"]["instrument_set"] == ["AAA", "BBB"]


def test_defensive_engine_error_uses_real_tickers_not_a_display_label(tmp_path):
    # (A)'s second half: `_record_defensive_outcomes` calls the SAME
    # `_instrument_set_for` helper, so a display-label `ticker` would raise
    # a SECOND time there, before it even manages to record a defensive
    # outcome -- the failure mode this test guards against.
    ledger = str(tmp_path / "ledger.jsonl")
    items = [_computed("a", ticker="a label, not a ticker", tickers=["AAA"])]

    def always_raise(**_kwargs):
        raise RuntimeError("boom")
    import unittest.mock
    with unittest.mock.patch.object(batch3.dsr, "deflated_sharpe_ratio", always_raise):
        with pytest.raises(RuntimeError, match="boom"):
            batch3.score_batch(items, ledger_path=ledger, run_id="run",
                              tested_date="2024-12-31", dry_run=False)
    outcome = next(e for e in trial_ledger.read_all(ledger) if e["event"] == "outcome")
    assert outcome["stage"] == "not_evaluated_error"
    assert outcome["evidence"]["observed"]["instrument_set"] == ["AAA"]


def test_insufficient_evidence_records_raw_catalog_parameters_not_the_adapter_view(tmp_path):
    # (B): `catalog_parameters()` is a factory-ADAPTER view -- it
    # renames/drops/casts keys per family (e.g. monthly_opex_week_drift
    # renames entry_trading_days_before_opex_friday to
    # entry_trading_days_before_opex). Recording THAT as the evidence
    # payload's observed.parameters would misstate the tested hypothesis
    # and later make validate_retry_request falsely report drift against
    # an unchanged real catalog entry. The RAW dict must be recorded
    # instead.
    ledger = str(tmp_path / "ledger.jsonl")
    raw_params = {"entry_trading_days_before_opex_friday": 3, "unrelated_key": "x"}
    item = _computed(trade_count=constants.MIN_TRADES_FOR_SCORING - 1,
                     raw_catalog_parameters=raw_params)
    result = batch3.score_batch(
        [item], ledger_path=ledger, run_id="run", tested_date="2024-12-31",
        dry_run=False)[0]
    assert result["stage"] == "insufficient_evidence"
    outcome = next(e for e in trial_ledger.read_all(ledger) if e["event"] == "outcome")
    assert outcome["evidence"]["observed"]["parameters"] == raw_params


def test_holdout_trade_count_below_the_floor_defers_a_dsr_passing_family(tmp_path):
    """A family that clears the in-sample DSR bar but whose holdout has
    fewer than MIN_HOLDOUT_TRADES trades must be deferred as
    insufficient evidence -- never scored via holdout_passes, and never
    silently recorded as a holdout cut or pass."""
    item = _computed("family-a", trade_count=200, sharpe=0.9)
    item["stats_holdout"] = {"trade_count": constants.MIN_HOLDOUT_TRADES - 1,
                             "net_edge_bps": 50.0, "sharpe": 0.3,
                             "skew": 0.0, "kurtosis": 3.0}
    item["holdout_net_edge_bps"] = 50.0
    correction = {"n_trials": 1.0, "sr_variance": 0.0004}
    result = batch3._score_one(item, correction)
    assert result["stage"] == "insufficient_evidence"
    assert result["dsr_value"] is None
    assert result["evidence"]["reason_code"] == "insufficient_holdout_trades"
    assert result["evidence"]["observed"]["holdout_trade_count"] == \
        constants.MIN_HOLDOUT_TRADES - 1
    assert result["evidence"]["observed"]["in_sample_trade_count"] == 200


def test_holdout_trade_count_at_the_floor_still_reaches_holdout_passes(tmp_path):
    """The boundary: exactly MIN_HOLDOUT_TRADES holdout trades is SCOREABLE
    (house >=-passes convention, matching MIN_TRADES_FOR_SCORING's own
    boundary), so this family must reach the holdout verdict (cut or pass), never the
    deferral path."""
    item = _computed("family-a", trade_count=200, sharpe=0.9)
    item["stats_holdout"] = {"trade_count": constants.MIN_HOLDOUT_TRADES,
                             "net_edge_bps": -5.0, "sharpe": -0.1,
                             "skew": 0.0, "kurtosis": 3.0}
    item["holdout_net_edge_bps"] = -5.0
    correction = {"n_trials": 1.0, "sr_variance": 0.0004}
    result = batch3._score_one(item, correction)
    assert result["stage"] in ("cut_holdout", "survived")
    assert result["dsr_value"] is not None


def test_holdout_insufficiency_is_checked_only_after_the_dsr_bar(tmp_path):
    """A family that fails the DSR bar outright must still be a DSR cut, even
    if its holdout ALSO has too few trades -- the holdout floor is a
    holdout-stage gate and must never run before the DSR bar has passed."""
    item = _computed("family-a", trade_count=200, sharpe=0.01)
    item["stats_holdout"] = {"trade_count": 1, "net_edge_bps": 10.0,
                             "sharpe": 0.1, "skew": 0.0, "kurtosis": 3.0}
    item["holdout_net_edge_bps"] = 10.0
    correction = {"n_trials": 1.0, "sr_variance": 0.0004}
    result = batch3._score_one(item, correction)
    assert result["stage"] == "cut_dsr"
    assert result["dsr_value"] is not None


def test_compute_candidate_carries_real_tickers_and_raw_catalog_parameters(
        tmp_path, monkeypatch):
    # Closes (A)/(B) at the SOURCE, not just at _score_one's consumption
    # side: compute_candidate's own result dict must carry the real traded
    # ticker list and the unmapped catalog parameters dict.
    # validate_supported_contract pins a real mechanism-text SHA-256 this
    # unit test has no need to reproduce -- it is already covered by
    # test_supported_contract_drift_fails_closed and friends; this test is
    # only about compute_candidate's own result-dict shape.
    monkeypatch.setattr(batch3, "validate_supported_contract", lambda *a, **k: None)
    frame = _frame(n=40)
    frames = {"COWZ": frame}
    raw_params = {"screen_metric": "free_cash_flow_yield", "portfolio_size": 100,
                 "rebalance": "quarterly"}
    candidate = {
        "family_id": "free_cash_flow_yield_cows",
        "catalog": {"family_id": "free_cash_flow_yield_cows",
                   "tickers": ["COWZ", "DSTL"],
                   "account_types": ["taxable", "roth"],
                   "parameters": raw_params},
        "matrix": {"mechanism_class": "single_primary_proxy_buy_and_hold",
                  "catalog_symbols": ["COWZ", "DSTL"],
                  "required_traded_symbols": ["COWZ"], "signal_symbols": [],
                  "reason": "test"},
        # Bypasses selected_execution_costs' own quote-selection machinery
        # (irrelevant to this test's purpose) the same way real callers can
        # via this candidate-level override.
        "execution_costs": {"COWZ": {"cost_bps": 5.0}},
    }
    window = {"holdout_start": str(frame.index[25].date()),
             "holdout_end": str(frame.index[-2].date())}
    result = batch3.compute_candidate(candidate, frames, None, window)
    assert result["tickers"] == ["COWZ"]
    assert result["raw_catalog_parameters"] == raw_params


def test_score_batch_allows_retrying_a_not_evaluated_error_family_at_the_same_window(
        tmp_path):
    # (C): a not_evaluated_error family is a caught DEFECT, not
    # evidence about the market (evidence.build_engine_error's own
    # docstring) -- it must be able to re-enter scoring once the defect is
    # fixed, unlike a real outcome. Safe only against the family's OWN
    # already-pinned window (record_first_test is idempotent and will not
    # re-pin), which this test deliberately reuses.
    ledger = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_first_test(
        ledger, "family-a", "2024-12-01", "2024-07-01", "2024-12-31", [], "run-0")
    trial_ledger.record_outcome(
        ledger, "family-a", "run-0", "2024-12-01", "not_evaluated_error",
        dsr_value=None,
        evidence=batch3.evidence.build_engine_error(
            error=RuntimeError("earlier bug"), instrument_set=["SPY"],
            pinned_window={"holdout_start": "2024-07-01", "holdout_end": "2024-12-31"},
            holdout_look="unspent"))
    # Same holdout_start/holdout_end as what is already pinned above.
    item = _computed(trade_count=constants.MIN_TRADES_FOR_SCORING - 1)
    result = batch3.score_batch(
        [item], ledger_path=ledger, run_id="run-1", tested_date="2024-12-31",
        dry_run=False)[0]
    assert result["stage"] == "insufficient_evidence"
    events = trial_ledger.read_all(ledger)
    assert len([e for e in events if e["event"] == "first_test"]) == 1  # not re-pinned
    assert len([e for e in events if e["event"] == "outcome"]) == 2  # the retry's new outcome


def test_score_batch_refuses_a_not_evaluated_error_retry_at_a_different_window(tmp_path):
    # The other half of (C): retrying is safe ONLY against the
    # SAME pinned window. A different window here would silently record an
    # outcome that does not match what the ledger says was tested --
    # refuse loudly instead.
    ledger = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_first_test(
        ledger, "family-a", "2024-06-01", "2024-01-01", "2024-06-30", [], "run-0")
    trial_ledger.record_outcome(
        ledger, "family-a", "run-0", "2024-06-01", "not_evaluated_error",
        dsr_value=None,
        evidence=batch3.evidence.build_engine_error(
            error=RuntimeError("earlier bug"), instrument_set=["SPY"],
            pinned_window={"holdout_start": "2024-01-01", "holdout_end": "2024-06-30"},
            holdout_look="unspent"))
    # _computed()'s default window (2024-07-01..2024-12-31) does NOT match
    # the pinned one above (2024-01-01..2024-06-30).
    item = _computed(trade_count=constants.MIN_TRADES_FOR_SCORING - 1)
    with pytest.raises(RuntimeError, match="window mismatch"):
        batch3.score_batch(
            [item], ledger_path=ledger, run_id="run-1", tested_date="2024-12-31",
            dry_run=False)


def test_score_batch_still_refuses_to_retry_a_real_outcome(tmp_path):
    # Regression guard: (C) must NOT relax the guard for a
    # family that genuinely already has an outcome (any DSR cut, holdout
    # cut, pass, or insufficient-evidence stage) -- only not_evaluated_error is a
    # defect, not a real prior test.
    ledger = str(tmp_path / "ledger.jsonl")
    trial_ledger.record_first_test(
        ledger, "family-a", "2024-12-01", "2024-07-01", "2024-12-31", [], "run-0")
    trial_ledger.record_outcome(
        ledger, "family-a", "run-0", "2024-12-01", "cut_dsr", dsr_value=0.4)
    item = _computed(trade_count=constants.MIN_TRADES_FOR_SCORING - 1)
    with pytest.raises(RuntimeError, match="already pinned"):
        batch3.score_batch(
            [item], ledger_path=ledger, run_id="run-1", tested_date="2024-12-31",
            dry_run=False)


@pytest.mark.parametrize("field,bad_value", [
    ("source_url", "https://example.invalid/finra.xlsx"),
    ("worksheet", "Other sheet"),
    ("units", "USD"),
    ("date_semantics", "publication_date"),
    ("retrieved_at", "not-a-timestamp"),
    ("retrieved_at", "2026-01-15T12:00:00"),
    ("raw_sha256", "abc"),
    ("raw_sha256", "g" * 64),
])
def test_finra_provenance_contract_fails_closed(field, bad_value):
    metadata = _valid_finra_metadata()
    metadata[field] = bad_value
    snapshot = batch3.external_data.ExternalSeriesSnapshot(
        pd.Series([1.0], index=[pd.Timestamp("2026-07-31")]), metadata)

    with pytest.raises(ValueError, match=field):
        batch3.validate_finra_snapshot(snapshot)


def test_json_projection_never_leaks_pandas_objects():
    projected = batch3.public_result(_computed())
    json.dumps(projected)
    assert "daily_returns" not in projected


def test_normalize_ohlc_require_positive_false_allows_a_negative_close():
    """A signal-only external series (e.g. ^IRX, which had real negative
    closes in March 2020) is never traded/priced -- require_positive=False
    lets a caller that KNOWS a symbol is signal-only opt out of the
    positivity gate. The default (require_positive=True) still rejects a
    negative close -- see the next test."""
    raw = _frame(n=5).reset_index().rename(columns={"index": "Date"})
    raw.loc[2, "Close"] = -0.05
    raw.loc[2, "Open"] = -0.04
    normalized = batch3._normalize_ohlc(raw, "^IRX", require_positive=False)
    assert normalized["Close"].iloc[2] == -0.05


def test_normalize_ohlc_default_still_rejects_a_negative_close():
    raw = _frame(n=5).reset_index().rename(columns={"index": "Date"})
    raw.loc[2, "Close"] = -0.05
    with pytest.raises(ValueError, match="positive"):
        batch3._normalize_ohlc(raw, "SPY")


def test_normalize_ohlc_require_positive_false_still_rejects_non_finite():
    raw = _frame(n=5).reset_index().rename(columns={"index": "Date"})
    raw.loc[2, "High"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        batch3._normalize_ohlc(raw, "^IRX", require_positive=False)
