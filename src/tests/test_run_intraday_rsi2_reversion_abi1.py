"""Driver-level tests for scripts/run_intraday_rsi2_reversion_abi1.py -- the
bespoke ABI-1 driver for intraday_rsi2_reversion_daytrade (SPY, long side).

The driver scores a day-trading family under the ABI-1 account-relative
construction.

Three groups of tests:

1. Registration / parameter pinning -- the driver must score the family's
   REGISTERED parameters, passed to the signal factory EXPLICITLY (never via
   the factory's own defaults: a silent default is exactly the drift the
   catalog check exists to prevent), and must refuse to run when the factory
   grows a parameter the driver does not pin.
2. Signal behaviour through the driver's own ``run_real_intraday_backtest``
   on small hand-built 1-minute sessions: entry timing, exit rules, the
   09:30 RSI reset (RSI must never carry across sessions), and causality
   (the entry decision at bar t must not read bar t+1).
3. The irreversible-path guards shared with every sibling intraday ABI-1
   driver (a dedicated unparametrized copy, same hermetic harness as
   test_run_green_day_morning_pullback_abi1.py; this driver is not in
   test_intraday_abi1_driver_guards.py's matrix).

Every test is hermetic -- the ledger, catalog, quotes file and research root
all live under ``tmp_path`` and the network is monkeypatched; the REAL trial
ledger is never touched -- with one disclosed exception: two catalog
consistency tests READ (never write) the tracked data/mining/catalog.json to
prove the driver still matches the registered definition.
"""
import inspect
import json
import os
import sys
import threading
import time

import numpy as np
import pandas as pd
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
for _path in (_SRC, _SCRIPTS, os.path.dirname(__file__)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_intraday_rsi2_reversion_abi1 as driver  # noqa: E402
from _intraday_abi1_driver_support import (Harness, daily_benchmark_frame, minute_bar_frame,  # noqa: E402
                                           write_quotes)
from mining import catalog as catalog_mod  # noqa: E402

TZ = "America/New_York"
TESTED_DATE = "2026-09-18"   # the shared harness's default sampled/tested date
QUOTE_SYMBOLS = ("SPY",)
TICKERS = ["SPY"]
FACTORY_NAME = "make_intraday_rsi2_reversion_daytrade_signal_fn"
REAL_CATALOG = os.path.join(os.path.dirname(_SRC), "data", "mining", "catalog.json")


class RsiHarness(Harness):
    """The shared harness plus this driver's real-run acknowledgements: a real
    (non-dry) run must name its tested date and accept that only the LONG side
    is scored. Tests that exercise the refusal itself override them."""

    def __init__(self, tmp_path, monkeypatch, driver_module, **kw):
        super().__init__(tmp_path, monkeypatch, driver_module, **kw)
        # A real run must be dated TODAY (ET); pin "today" to the harness's
        # tested date so the rule is deterministic.
        monkeypatch.setattr(driver_module, "_today_et", lambda: TESTED_DATE)

        # A real run refuses when the LAST session is incomplete (a partial day
        # cached mid-session), so the default frame ends on a full, flat
        # 390-bar session (flat: RSI 50, never an entry).
        def fetch_with_a_complete_last_session(symbol, start_date, end_date, **_kwargs):
            self.fetch_calls.append((symbol, start_date, end_date))
            return pd.concat([minute_bar_frame(days=("2026-03-02", "2026-03-03")),
                              _session("2026-03-04", [100.0] * 78)])

        monkeypatch.setattr(driver_module.intraday_data, "fetch_intraday_bars",
                            fetch_with_a_complete_last_session)

    def kwargs(self, **overrides):
        base = super().kwargs(**overrides)
        base.setdefault("accept_long_side_only", True)
        return base

    def cli_argv(self, *extra):
        return super().cli_argv("--tested-date", TESTED_DATE, "--accept-long-side-only", *extra)


def _rewrite_quotes(harness, *, sampled_at_local=None, market_state=None):
    with open(harness.quotes_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if sampled_at_local is not None:
        payload["sampled_at_local"] = sampled_at_local
    if market_state is not None:
        payload["market_state"] = market_state
    with open(harness.quotes_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


# ------------------------------------------------------------- session builders

def _session(date: str, bucket_prices) -> pd.DataFrame:
    """An RTH session from 09:30 (78 buckets = the full 390 bars to 15:59; 42 =
    an early-close half day to 12:59). Bucket k is the k-th 5-minute
    block (09:30-09:34 is bucket 0); every bar of a bucket sits at that
    bucket's price, so the 5-minute close the signal sees IS the bucket price
    and every bar's Open equals its bucket price."""
    prices = np.repeat(np.asarray(bucket_prices, dtype=float), 5)
    index = pd.date_range(pd.Timestamp(f"{date} 09:30", tz=TZ), periods=len(prices), freq="1min")
    return pd.DataFrame(
        {"Open": prices, "High": prices, "Low": prices, "Close": prices,
         "Volume": 1000.0, "TradeCount": 10, "VWAP": prices}, index=index)


def _dip_and_recover_prices():
    """Flat 100 to 10:09, two down buckets (99, 98) then a bounce to 99.5:
    RSI(2) is oversold from the first down bucket (signal bar 10:14), the
    engine fills at the NEXT bar's open (10:15, price 98), and the 5-bar-SMA
    cross-back fires at the 10:24 bar (close 99.5)."""
    return [100.0] * 8 + [99.0, 98.0, 99.5] + [99.5] * (78 - 11)


def _dip_session(date="2026-03-02") -> pd.DataFrame:
    return _session(date, _dip_and_recover_prices())


def _gap_down_then_flat() -> pd.DataFrame:
    """Day 1 sits flat at 100 until its very last bucket (15:55-15:59), which
    prints 90 -- no entry is possible there (force-flat guard). Day 2 opens at
    that gap-down price and stays flat at 90 all session. A fresh-at-09:30
    RSI(2) sees an all-flat day (RSI 50, never oversold); an RSI that carried
    yesterday's closes across the boundary sees a -10 move and reads deeply
    oversold for the whole morning."""
    return pd.concat([
        _session("2026-03-02", [100.0] * 77 + [90.0]),
        _session("2026-03-03", [90.0] * 78),
    ])


def _wilder_rsi(closes, period=2):
    """Independent Wilder RSI, used only to show WHY the gap frame
    discriminates carried-vs-fresh RSI (not to compute expectations for the
    driver's own output)."""
    closes = pd.Series(list(closes), dtype=float)
    diffs = closes.diff().dropna()
    gains, losses = diffs.clip(lower=0.0), (-diffs).clip(lower=0.0)
    avg_gain, avg_loss = gains.iloc[:period].mean(), losses.iloc[:period].mean()
    for g, l in zip(gains.iloc[period:], losses.iloc[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def _spy_factory(monkeypatch):
    """Wrap the real factory so every call's EXPLICIT kwargs are recorded.
    The wrapper takes **kwargs only, so a parameter the driver leaves out is
    absent from what is recorded and silently filled by the real factory's own
    default -- exactly the omission this test group is meant to catch."""
    real = getattr(driver.mod, FACTORY_NAME)
    calls = []

    def spy(**kwargs):
        signal_fn = real(**kwargs)
        calls.append((dict(kwargs), signal_fn))
        return signal_fn

    monkeypatch.setattr(driver.mod, FACTORY_NAME, spy)
    return real, calls


# ================================================== 1. registration / pinning

def test_driver_parameters_match_the_real_catalog_entry():
    entry = {e["family_id"]: e for e in catalog_mod.load_catalog(REAL_CATALOG)}[driver.FAMILY_ID]
    assert driver._registration_drift(entry) is None
    assert entry["tickers"] == TICKERS
    assert entry["timeframe"] == "day"


def test_signal_is_built_from_the_registered_parameters_not_callee_defaults(monkeypatch):
    real, calls = _spy_factory(monkeypatch)
    driver.run_real_intraday_backtest(_dip_session(), cost_bps=0.0)

    assert len(calls) == 1
    passed, _fn = calls[0]
    # NOTHING left to the factory's own defaults: every parameter it accepts
    # was passed explicitly by the driver.
    assert set(passed) == set(inspect.signature(real).parameters)

    catalog_parameters = {e["family_id"]: e for e in catalog_mod.load_catalog(REAL_CATALOG)
                          }[driver.FAMILY_ID]["parameters"]
    for name, value in passed.items():
        if name in catalog_parameters:
            assert value == catalog_parameters[name], name
    # sma_period is not a catalog key: the registered fact is the exit_rule
    # name "5bar_sma_cross_or_force_flat", and the driver derives it from that.
    assert passed["sma_period"] == 5
    assert catalog_parameters["exit_rule"].startswith("5bar_")


def test_sma_period_is_derived_from_the_registered_exit_rule_name():
    assert driver.sma_period_from_exit_rule("5bar_sma_cross_or_force_flat") == 5
    assert driver.sma_period_from_exit_rule("8bar_sma_cross_or_force_flat") == 8
    assert driver.SMA_PERIOD == driver.sma_period_from_exit_rule(driver.PARAMETERS["exit_rule"])
    with pytest.raises(ValueError, match="exit_rule"):
        driver.sma_period_from_exit_rule("trailing_stop_or_force_flat")


def test_a_factory_parameter_the_driver_does_not_pin_blocks_a_run(tmp_path, monkeypatch):
    """If the factory grows a new keyword (with a default), the driver would
    silently score it at that default. The run must refuse instead, before
    any fetch."""
    real = getattr(driver.mod, FACTORY_NAME)

    def grown(rsi_period=2, bar_minutes=5, oversold_level=10.0, overbought_level=90.0,
              earliest_entry_et="10:00", force_flat_et="15:55", sma_period=5, new_knob=3):
        return real()

    harness = RsiHarness(tmp_path, monkeypatch, driver,
                      quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    monkeypatch.setattr(driver.mod, FACTORY_NAME, grown)
    result = driver.compute_and_record(**harness.kwargs())
    assert result["reason_code"] == "parameter_drift"
    assert "new_knob" in result["reason"]
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []


# ============================================== 2. signal behaviour via driver

def test_an_oversold_dip_enters_at_the_next_bars_open_and_exits_on_the_sma_cross_back():
    trades = driver.run_real_intraday_backtest(_dip_session(), cost_bps=0.0)
    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["entry_datetime"] == pd.Timestamp("2026-03-02 10:15", tz=TZ)
    # Filled at the NEXT bar's open (98), not the signal bar's close (99).
    assert row["entry_px"] == pytest.approx(98.0)
    assert row["exit_datetime"] == pd.Timestamp("2026-03-02 10:24", tz=TZ)
    assert row["exit_px"] == pytest.approx(99.5)
    assert row["reason"] == "signal_exit"
    assert row["gross_ret"] == pytest.approx(99.5 / 98.0 - 1.0)


def test_no_entry_is_taken_before_the_registered_earliest_entry_time():
    """RSI(2) is already oversold from 09:49 on this steadily falling tape,
    but the registered earliest entry is 10:00: the first decision bar at or
    after it is the 10:04 bucket end, so the fill is the 10:05 bar's open."""
    prices = [100.0] * 3 + [99.0, 98.0, 97.0, 96.0] + [95.0] * (78 - 7)
    trades = driver.run_real_intraday_backtest(_session("2026-03-02", prices), cost_bps=0.0)
    assert len(trades) >= 1
    first = trades.iloc[0]
    assert first["entry_datetime"] == pd.Timestamp("2026-03-02 10:05", tz=TZ)
    assert first["entry_px"] == pytest.approx(95.0)
    # The tape then goes flat at 95: the 5-bar SMA equals the last close at the
    # 10:29 bucket end and the exit is a `close >= SMA` LEVEL check (equality
    # exits) -- not a strict cross, which would ride to the 15:55 force-flat.
    assert first["exit_datetime"] == pd.Timestamp("2026-03-02 10:29", tz=TZ)
    assert first["exit_px"] == pytest.approx(95.0)


def test_a_position_still_falling_at_1555_is_force_flattened_that_bar():
    prices = [100.0] * 66 + [100.0 - 0.5 * (k + 1) for k in range(12)]
    trades = driver.run_real_intraday_backtest(_session("2026-03-02", prices), cost_bps=0.0)
    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["entry_datetime"] == pd.Timestamp("2026-03-02 15:05", tz=TZ)
    assert row["exit_datetime"] == pd.Timestamp("2026-03-02 15:55", tz=TZ)
    assert row["exit_px"] == pytest.approx(94.0)
    assert row["held_days"] == 1


def test_rsi_never_carries_across_the_session_boundary():
    bars = _gap_down_then_flat()
    # Why this frame discriminates: fed yesterday's closes the RSI reads
    # deeply oversold on day 2; fed only today's it reads a neutral 50.
    day1_closes = [100.0] * 77 + [90.0]
    assert _wilder_rsi(day1_closes + [90.0] * 8) < 10.0
    assert _wilder_rsi([90.0] * 8) == 50.0

    trades = driver.run_real_intraday_backtest(bars, cost_bps=0.0)
    assert len(trades) == 0, (
        "an entry on day 2 means RSI(2) was seeded with day 1's closes -- the "
        "mechanism resets it fresh at 09:30 every session")


def _scramble_after_signal_bar(bars: pd.DataFrame, fill_ts, *, level=None, seed=None) -> pd.DataFrame:
    """Scramble everything the engine may legitimately NOT have known at the
    signal bar: the fill bar's High/Low/Close and every later bar. The fill
    bar's OPEN is kept -- the engine fills there by contract. `level` sets all
    of it to one constant (a crash or a spike); `seed` makes a random walk."""
    out = bars.copy()
    later = out.index > fill_ts
    fill = out.index == fill_ts
    assert fill.sum() == 1 and later.sum() > 0
    if level is not None:
        for col in ("Open", "High", "Low", "Close"):
            out.loc[later, col] = level
        out.loc[fill, ["High", "Low", "Close"]] = level
    else:
        rng = np.random.default_rng(seed)
        walk = np.exp(np.cumsum(rng.normal(0.0, 0.02, later.sum()))) * 98.0
        for col in ("Open", "High", "Low", "Close"):
            out.loc[later, col] = walk
        out.loc[fill, ["High", "Low", "Close"]] = 98.0 * float(np.exp(rng.normal(0.0, 0.05)))
    return out


def test_the_entry_decision_at_bar_t_never_reads_bar_t_plus_1():
    bars = _dip_session()
    fill_ts = pd.Timestamp("2026-03-02 10:15", tz=TZ)   # signal bar is 10:14
    baseline = driver.run_real_intraday_backtest(bars, cost_bps=0.0).iloc[0]
    assert baseline["entry_datetime"] == fill_ts

    variants = [
        _scramble_after_signal_bar(bars, fill_ts, level=1.0),      # crash
        _scramble_after_signal_bar(bars, fill_ts, level=1000.0),   # spike
    ] + [_scramble_after_signal_bar(bars, fill_ts, seed=seed) for seed in range(8)]
    # The data really was scrambled (guards against a vacuous perturbation).
    assert all(not v.loc[fill_ts:, "Close"].equals(bars.loc[fill_ts:, "Close"]) for v in variants)

    for variant in variants:
        trades = driver.run_real_intraday_backtest(variant, cost_bps=0.0)
        assert len(trades) >= 1
        first = trades.iloc[0]
        assert first["entry_datetime"] == baseline["entry_datetime"]
        assert first["entry_px"] == baseline["entry_px"] == pytest.approx(98.0)

    # And with NOTHING after the fill bar at all the same entry is taken.
    truncated = driver.run_real_intraday_backtest(bars.loc[:fill_ts], cost_bps=0.0)
    assert truncated.iloc[0]["entry_datetime"] == baseline["entry_datetime"]
    assert truncated.iloc[0]["entry_px"] == baseline["entry_px"]


def test_an_early_close_last_bar_signal_fills_at_the_next_sessions_open_and_is_counted():
    """The factory's own disclosed residual: on an early-close half day
    (last bar 12:59, well before the 15:55 force-flat) an oversold last bucket
    still fires, and the engine fills it at the FOLLOWING session's open --
    an entry priced off yesterday's RSI. The mechanism forbids any entry
    before 10:00, so a 09:30 fill can only be this artifact; the driver
    counts them so the real run discloses how often it happened."""
    half_day = _session("2026-03-02", [100.0] * 41 + [99.0])   # first down bucket is the LAST one
    next_day = _session("2026-03-03", [99.0] * 78)
    trades = driver.run_real_intraday_backtest(pd.concat([half_day, next_day]), cost_bps=0.0)
    assert len(trades) == 1
    assert trades.iloc[0]["entry_datetime"] == pd.Timestamp("2026-03-03 09:30", tz=TZ)
    assert driver.count_session_open_entries(trades) == 1
    # A normal intraday entry is not counted, nor is an empty frame.
    normal = driver.run_real_intraday_backtest(_dip_session(), cost_bps=0.0)
    assert driver.count_session_open_entries(normal) == 0
    assert driver.count_session_open_entries(normal.iloc[0:0]) == 0


def test_a_fresh_signal_fn_instance_is_built_per_call(monkeypatch):
    _real, calls = _spy_factory(monkeypatch)
    bars = _dip_session()
    first = driver.run_real_intraday_backtest(bars, cost_bps=0.0)
    second = driver.run_real_intraday_backtest(bars, cost_bps=0.0)
    assert len(calls) == 2
    assert calls[0][1] is not calls[1][1]
    assert first.iloc[0]["net_ret"] == pytest.approx(second.iloc[0]["net_ret"])


def test_the_round_trip_cost_is_charged_exactly_once_per_trade():
    free = driver.run_real_intraday_backtest(_dip_session(), cost_bps=0.0).iloc[0]
    costed = driver.run_real_intraday_backtest(_dip_session(), cost_bps=2.0).iloc[0]
    assert costed["gross_ret"] == pytest.approx(free["gross_ret"])
    assert costed["net_ret"] == pytest.approx(free["net_ret"] - 2.0 / 1e4)


def test_run_real_intraday_backtest_never_applies_a_tax_haircut_at_the_engine_layer():
    """ABI-1's own tax model applies tax exactly once, from PRE-tax net_ret; a
    nonzero engine haircut would double-tax every winner (the vwap sibling's
    test file guards the same rule)."""
    trades = driver.run_real_intraday_backtest(_dip_session(), cost_bps=0.0)
    row = trades.iloc[0]
    assert row["net_ret"] > 0.0
    assert row["net_ret"] == row["gross_ret"]
    with pytest.raises(TypeError):
        driver.run_real_intraday_backtest(_dip_session(), cost_bps=0.0, tax_haircut_rate=0.05)


def test_the_engine_is_run_long_only_with_the_registered_max_hold_days(monkeypatch):
    seen = {}
    real_engine = driver.run_intraday_backtest

    def spy_engine(df, signal_fn, **kwargs):
        seen.update(kwargs)
        return real_engine(df, signal_fn, **kwargs)

    monkeypatch.setattr(driver, "run_intraday_backtest", spy_engine)
    driver.run_real_intraday_backtest(_dip_session(), cost_bps=0.0)
    assert seen["direction"] == "long"
    assert seen["max_hold_days"] == driver.PARAMETERS["max_hold_days"] == 1
    assert seen["tax_haircut_rate"] == 0.0


# ======================================================== 3. end-to-end wiring

def test_a_dry_run_wires_real_signal_engine_bridge_and_evaluation_end_to_end(tmp_path, monkeypatch):
    bars = pd.concat([
        _dip_session("2026-03-02"),
        _session("2026-03-03", [100.0] * 78),   # a day with no trade
        _dip_session("2026-03-04"),
    ])
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                      quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)

    def fake_fetch(symbol, start_date, end_date, **_kwargs):
        harness.fetch_calls.append((symbol, start_date, end_date))
        return bars

    monkeypatch.setattr(driver.intraday_data, "fetch_intraday_bars", fake_fetch)
    write_quotes(harness.quotes_path, QUOTE_SYMBOLS, synthetic=True)

    seen = {}
    real_backtest = driver.run_real_intraday_backtest
    real_evaluate = driver.abi1_driver.evaluate_family

    def spy_backtest(fetched_bars, *, cost_bps):
        seen["cost_bps"] = cost_bps
        return real_backtest(fetched_bars, cost_bps=cost_bps)

    def spy_evaluate(curve, **kwargs):
        seen["benchmark_entry_cost_bps"] = kwargs["benchmark_entry_cost_bps"]
        seen["curve_days"] = list(curve["strategy_daily"].index)
        seen["curve_returns"] = list(curve["strategy_daily"].values)
        seen["tax_policy"] = kwargs["tax_policy"]
        seen["account_type"] = kwargs["account_type"]
        return real_evaluate(curve, **kwargs)

    monkeypatch.setattr(driver, "run_real_intraday_backtest", spy_backtest)
    monkeypatch.setattr(driver.abi1_driver, "evaluate_family", spy_evaluate)

    result = driver.compute_and_record(**harness.kwargs(dry_run=True))

    assert "reason_code" not in result
    assert result["tickers"] == TICKERS
    assert result["n_trades"] == 2
    assert result["n_intraday_sessions"] == 3
    assert result["entries_filled_at_session_open"] == 0
    # The quote (0.06 wide on a 600.03 mid = ~1.00 bps) prices the round trip
    # in full and the benchmark's entry at HALF the spread.
    assert seen["cost_bps"] == pytest.approx(1.0, abs=1e-3)
    assert seen["benchmark_entry_cost_bps"] == pytest.approx(0.5, abs=1e-3)
    assert seen["tax_policy"] == "realization" and seen["account_type"] == "taxable"
    # The bridge produced one daily row per real session: trade days carry the
    # trade's own net return, the no-trade day is exactly 0.
    assert seen["curve_days"] == [pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-03"),
                                  pd.Timestamp("2026-03-04")]
    expected = 99.5 / 98.0 - 1.0 - 1.0 / 1e4
    assert seen["curve_returns"][0] == pytest.approx(expected, abs=1e-6)
    assert seen["curve_returns"][1] == 0.0
    assert seen["curve_returns"][2] == pytest.approx(expected, abs=1e-6)
    assert result["real_intraday_data_start"] == driver.REAL_DATA_START
    # cost comes from the quotes file: 0.06 wide on 600.03 mid = 1.00 bps.
    assert "SPY=1.00bps" in result["cost_model_note"]
    assert "long side only" in result["cost_model_note"]
    assert result["direction"] == "long"
    assert set(result["timing_seconds"]) == {"fetch", "backtest", "bridge_and_evaluate"}
    assert all(seconds >= 0.0 for seconds in result["timing_seconds"].values())
    assert harness.fetch_calls == [("SPY", driver.REAL_DATA_START, "2026-03-04")]
    assert harness.ledger_rows() == []


def test_a_dry_run_reports_the_session_open_fill_count_in_its_result(tmp_path, monkeypatch):
    half_day = _session("2026-03-02", [100.0] * 41 + [99.0])
    bars = pd.concat([half_day, _session("2026-03-03", [99.0] * 78),
                      _session("2026-03-04", [100.0] * 78)])
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                      quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    monkeypatch.setattr(driver.intraday_data, "fetch_intraday_bars",
                        lambda symbol, start_date, end_date, **_kw: bars)
    write_quotes(harness.quotes_path, QUOTE_SYMBOLS, synthetic=True)
    result = driver.compute_and_record(**harness.kwargs(dry_run=True))
    assert result["n_trades"] == 1
    assert result["entries_filled_at_session_open"] == 1


# ------------------------------------------------ real-run window overrides

def test_real_run_refuses_an_end_date_override(tmp_path, monkeypatch):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                      quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    with pytest.raises(ValueError, match="end_date"):
        driver.compute_and_record(**harness.kwargs(end_date="2026-03-03"))
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []


def test_dry_run_still_accepts_an_end_date_override(tmp_path, monkeypatch):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                      quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    result = driver.compute_and_record(**harness.kwargs(dry_run=True, end_date="2026-03-03"))
    assert "reason_code" not in result
    assert harness.fetch_calls
    assert all(end == "2026-03-03" for _symbol, _start, end in harness.fetch_calls)
    assert harness.ledger_rows() == []


def test_real_run_refuses_a_start_date_override(tmp_path, monkeypatch):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                      quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    with pytest.raises(ValueError, match="start_date"):
        driver.compute_and_record(**harness.kwargs(start_date="2026-03-03"))
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []


def test_dry_run_still_accepts_a_start_date_override(tmp_path, monkeypatch):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                      quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    result = driver.compute_and_record(**harness.kwargs(dry_run=True, start_date="2026-03-02"))
    assert "reason_code" not in result
    assert all(start == "2026-03-02" for _symbol, start, _end in harness.fetch_calls)


def test_a_real_run_always_fetches_from_the_measured_floor(tmp_path, monkeypatch):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                      quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    driver.compute_and_record(**harness.kwargs())
    assert harness.fetch_calls
    assert all(start == driver.REAL_DATA_START for _symbol, start, _end in harness.fetch_calls)


# ----------------------------------------------------------- quotes guard

def test_real_run_refuses_a_synthetic_quotes_file(tmp_path, monkeypatch):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                      quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    write_quotes(harness.quotes_path, QUOTE_SYMBOLS, synthetic=True)
    with pytest.raises(ValueError, match="SYNTHETIC"):
        driver.compute_and_record(**harness.kwargs())
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []


def test_a_quotes_file_without_a_spy_quote_refuses_before_any_fetch(tmp_path, monkeypatch):
    """SPY is both the traded instrument and the ABI-1 benchmark: its live
    quote prices the strategy's round-trip cost AND the benchmark's entry
    cost. No SPY quote means no honest score."""
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                      quote_symbols=("QQQ",), catalog_tickers=TICKERS)
    with pytest.raises(ValueError, match="SPY"):
        driver.compute_and_record(**harness.kwargs())
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []


# ------------------------------------------------- ledger side effects

def test_a_dry_run_writes_no_ledger_rows(tmp_path, monkeypatch):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                      quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    write_quotes(harness.quotes_path, QUOTE_SYMBOLS, synthetic=True)
    result = driver.compute_and_record(**harness.kwargs(dry_run=True))
    assert "reason_code" not in result
    assert harness.fetch_calls
    assert "verdict" in result and "stage" in result
    assert harness.ledger_rows() == []
    assert not os.path.exists(harness.ledger_path)


def test_a_real_run_writes_exactly_one_abi1_outcome_and_a_second_is_blocked(tmp_path, monkeypatch):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                      quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    first = driver.compute_and_record(**harness.kwargs())
    assert "reason_code" not in first
    rows_after_first = harness.ledger_rows()
    outcomes = [r for r in rows_after_first if r.get("event") == "outcome"]
    assert len(outcomes) == 1
    assert outcomes[0]["family_id"] == driver.FAMILY_ID
    assert outcomes[0]["construction"] == "ABI-1"

    calls_before = len(harness.fetch_calls)
    second = driver.compute_and_record(**harness.kwargs())
    assert second["reason_code"] == "already_scored_abi1"
    assert len(harness.fetch_calls) == calls_before
    assert harness.ledger_rows() == rows_after_first


def test_two_concurrent_real_runs_only_one_outcome_is_ever_recorded(tmp_path, monkeypatch):
    """The race trial_ledger.record_outcome_if_not_scored closes: the
    unlocked already_scored prefilter and the eventual append are two
    operations with real compute in between. Genuine threads, real compute
    through the real backtest+evaluate+ledger-write path -- not a mock."""
    from mining import trial_ledger

    harness = RsiHarness(tmp_path, monkeypatch, driver,
                      quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    original_read_all = trial_ledger.read_all

    def slow_read_all(path):
        result = original_read_all(path)
        time.sleep(0.15)
        return result

    monkeypatch.setattr(trial_ledger, "read_all", slow_read_all)

    results = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        result = driver.compute_and_record(**harness.kwargs())
        with results_lock:
            results.append(result)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    outcomes = [r for r in harness.ledger_rows() if r.get("event") == "outcome"]
    assert len(outcomes) == 1, (
        f"expected exactly one permanent ABI-1 outcome row after two "
        f"concurrent real runs, found {len(outcomes)}: {outcomes}")
    succeeded = [r for r in results if "reason_code" not in r]
    skipped = [r for r in results if "reason_code" in r]
    assert len(succeeded) == 1
    assert len(skipped) == 1
    assert skipped[0]["reason_code"] == "already_scored_abi1"


def test_a_seeded_abi1_outcome_blocks_a_real_run_before_any_fetch(tmp_path, monkeypatch):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                      quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    harness.seed_abi1_outcome()
    seeded = harness.ledger_rows()
    result = driver.compute_and_record(**harness.kwargs())
    assert result["reason_code"] == "already_scored_abi1"
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == seeded


def test_a_prior_per_trade_outcome_does_not_block_an_abi1_run(tmp_path, monkeypatch):
    """A family may carry a legacy per_trade outcome row from the older
    construction. The one-shot guard keys on construction == "ABI-1", so a
    legacy row must NOT block the family's first ABI-1 attempt: a dry run below
    is not refused."""
    from mining import trial_ledger

    harness = RsiHarness(tmp_path, monkeypatch, driver,
                      quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    trial_ledger.record_outcome(
        harness.ledger_path, driver.FAMILY_ID, "legacy-run", "2020-01-02",
        stage="cut_dsr", dsr_value=0.01, sharpe_in_sample=0.1,
        ticker_label="SPY", note="legacy per-trade outcome")   # construction=per_trade
    assert not any(r.get("construction") == "ABI-1" for r in harness.ledger_rows())
    result = driver.compute_and_record(**harness.kwargs(dry_run=True))
    assert "reason_code" not in result


# ----------------------------------------------------- catalog/registration

def test_a_non_eligible_catalog_status_skips_before_any_fetch(tmp_path, monkeypatch):
    harness = RsiHarness(tmp_path, monkeypatch, driver, quote_symbols=QUOTE_SYMBOLS,
                      catalog_tickers=TICKERS, status="ruled_out")
    result = driver.compute_and_record(**harness.kwargs())
    assert result["reason_code"] == "catalog_status"
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []


def test_a_catalog_parameter_drift_skips_before_any_fetch(tmp_path, monkeypatch):
    drifted = dict(driver.PARAMETERS)
    drifted["force_flat_et"] = "15:30"
    harness = RsiHarness(tmp_path, monkeypatch, driver, quote_symbols=QUOTE_SYMBOLS,
                      catalog_tickers=TICKERS, catalog_parameters=drifted)
    result = driver.compute_and_record(**harness.kwargs())
    assert result["reason_code"] == "parameter_drift"
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []


def test_a_catalog_exit_rule_drift_skips_before_any_fetch(tmp_path, monkeypatch):
    """exit_rule is the ONLY place the registered SMA length lives; a catalog
    edit to it must stop the run, not be silently ignored."""
    drifted = dict(driver.PARAMETERS)
    drifted["exit_rule"] = "8bar_sma_cross_or_force_flat"
    harness = RsiHarness(tmp_path, monkeypatch, driver, quote_symbols=QUOTE_SYMBOLS,
                      catalog_tickers=TICKERS, catalog_parameters=drifted)
    result = driver.compute_and_record(**harness.kwargs())
    assert result["reason_code"] == "parameter_drift"
    assert harness.fetch_calls == []


def test_a_catalog_ticker_drift_skips_before_any_fetch(tmp_path, monkeypatch):
    harness = RsiHarness(tmp_path, monkeypatch, driver, quote_symbols=QUOTE_SYMBOLS,
                      catalog_tickers=["SPY", "QQQ"])
    result = driver.compute_and_record(**harness.kwargs())
    assert result["reason_code"] == "parameter_drift"
    assert "tickers" in result["reason"]
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []


# ------------------------------------------------------------ main() writes

def test_main_refuses_a_real_run_end_date_before_writing_anything(tmp_path, monkeypatch):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                      quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    with pytest.raises(ValueError, match="end_date"):
        driver.main(harness.cli_argv("--end-date", "2026-03-03"))
    assert not os.path.exists(harness.results_dir())
    assert harness.ledger_rows() == []


def test_main_writes_the_results_file_for_a_real_run(tmp_path, monkeypatch, capsys):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                      quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    assert driver.main(harness.cli_argv()) == 0
    path = os.path.join(harness.results_dir(), "intraday-rsi2-reversion-results.json")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["dry_run"] is False
    assert payload["result"]["family_id"] == driver.FAMILY_ID
    assert "verdict" in payload["result"]


def test_main_never_overwrites_the_real_results_file_with_a_skip(tmp_path, monkeypatch, capsys):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                      quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    assert driver.main(harness.cli_argv()) == 0
    path = os.path.join(harness.results_dir(), "intraday-rsi2-reversion-results.json")
    with open(path, "rb") as handle:
        first_bytes = handle.read()
    capsys.readouterr()

    assert driver.main(harness.cli_argv()) == 0
    with open(path, "rb") as handle:
        assert handle.read() == first_bytes
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["result"]["reason_code"] == "already_scored_abi1"
    assert printed["output"] is None


def test_main_writes_no_results_file_when_the_result_is_a_skip(tmp_path, monkeypatch, capsys):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                      quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    harness.seed_abi1_outcome()
    assert driver.main(harness.cli_argv()) == 0
    assert not os.path.exists(
        os.path.join(harness.results_dir(), "intraday-rsi2-reversion-results.json"))
    assert not os.path.exists(
        os.path.join(harness.results_dir(), "intraday-rsi2-reversion-dry-run.json"))


def test_main_writes_the_dry_run_file_for_a_dry_run(tmp_path, monkeypatch, capsys):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                      quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    write_quotes(harness.quotes_path, QUOTE_SYMBOLS, synthetic=True)
    assert driver.main(harness.cli_argv("--dry-run")) == 0
    assert os.path.exists(
        os.path.join(harness.results_dir(), "intraday-rsi2-reversion-dry-run.json"))
    assert not os.path.exists(
        os.path.join(harness.results_dir(), "intraday-rsi2-reversion-results.json"))
    assert harness.ledger_rows() == []


# ============================================ 4. real-run preconditions

def test_a_real_run_refuses_without_the_long_side_only_acknowledgement(tmp_path, monkeypatch):
    """The catalog mechanism has a short side the factory does not implement;
    a real run spends the family's ONLY ABI-1 slot on the long side alone. That
    must be an explicit, deliberate choice at the real-run command line."""
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    with pytest.raises(ValueError, match="long side"):
        driver.compute_and_record(**harness.kwargs(accept_long_side_only=False))
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []


def test_a_dry_run_needs_no_long_side_acknowledgement(tmp_path, monkeypatch):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    result = driver.compute_and_record(**harness.kwargs(dry_run=True, accept_long_side_only=False))
    assert "reason_code" not in result


def test_main_refuses_a_real_run_without_the_acknowledgement_flag(tmp_path, monkeypatch):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    argv = [a for a in harness.cli_argv() if a != "--accept-long-side-only"]
    with pytest.raises(ValueError, match="long side"):
        driver.main(argv)
    assert not os.path.exists(harness.results_dir())
    assert harness.ledger_rows() == []


def test_a_real_run_requires_an_explicit_tested_date(tmp_path, monkeypatch):
    """The tested date is written into the permanent ledger row; a silently
    defaulted (stale) date must be impossible."""
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    # Match the SPECIFIC refusal: without it the quote-date check would also
    # mention "tested_date" and let a missing explicit check go unnoticed.
    with pytest.raises(ValueError, match="needs an explicit tested_date"):
        driver.compute_and_record(**harness.kwargs(tested_date=None))
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []


def test_a_dry_run_needs_no_tested_date(tmp_path, monkeypatch):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    result = driver.compute_and_record(**harness.kwargs(dry_run=True, tested_date=None))
    assert "reason_code" not in result


def test_real_run_refuses_a_quotes_sample_from_a_closed_market(tmp_path, monkeypatch):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    _rewrite_quotes(harness, market_state="closed (Friday 2024-05-03 after 16:00 ET)")
    with pytest.raises(ValueError, match="market_state"):
        driver.compute_and_record(**harness.kwargs())
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []


def test_real_run_refuses_a_quotes_sample_from_another_day_than_the_tested_date(tmp_path, monkeypatch):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    _rewrite_quotes(harness, sampled_at_local="2024-05-01T10:00:00")
    with pytest.raises(ValueError, match="sampled_at_local"):
        driver.compute_and_record(**harness.kwargs())
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []


def test_a_real_run_refuses_a_tested_date_that_is_not_today(tmp_path, monkeypatch):
    """Matching the quote's date to --tested-date is not enough: pairing an OLD
    quote with its own old date would satisfy that and still price the
    permanent verdict off stale costs. A real run is dated today (ET)."""
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    _rewrite_quotes(harness, sampled_at_local="2024-05-01T10:00:00")
    with pytest.raises(ValueError, match="not today"):
        driver.compute_and_record(**harness.kwargs(tested_date="2024-05-01"))
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []


@pytest.mark.parametrize("market_state", [
    "open (Wednesday 2024-05-01, regular session, ~4m after the 09:30 ET open)",
    "Open (regular session)",
    "OPEN",
    "  open",
])
def test_a_real_run_accepts_every_realistic_open_market_state(tmp_path, monkeypatch, market_state):
    """A capture file may spell it "open (<weekday> <date>, regular
    session, ...)": the freshness rule must not refuse a legitimate capture."""
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    _rewrite_quotes(harness, market_state=market_state)
    result = driver.compute_and_record(**harness.kwargs())
    assert "reason_code" not in result


def test_real_run_refuses_a_quotes_sample_from_the_same_month_but_another_day(tmp_path, monkeypatch):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    _rewrite_quotes(harness, sampled_at_local="2026-09-17T10:00:00")
    with pytest.raises(ValueError, match="sampled_at_local"):
        driver.compute_and_record(**harness.kwargs())
    assert harness.ledger_rows() == []


def test_a_real_run_never_reuses_a_benchmark_cache_left_by_another_day(tmp_path, monkeypatch):
    """yfinance's newest daily row is an in-progress bar when fetched during the
    session, and batch5.load_frame caches it. A cache from an earlier day (or a
    dry run) would be scored as if that row were final. A real run therefore
    reads/writes its own per-tested-date cache; a dry run keeps the shared one."""
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    seen = []
    frame = daily_benchmark_frame()

    def spy_load_frame(symbol, *, ohlc_loader, cache_dir, **_kw):
        seen.append(os.path.basename(cache_dir))
        return frame, "fetched"

    monkeypatch.setattr(driver.batch5, "load_frame", spy_load_frame)
    driver.compute_and_record(**harness.kwargs(dry_run=True))
    driver.compute_and_record(**harness.kwargs())
    assert seen == ["ohlc", f"ohlc-{TESTED_DATE}"]


def test_a_dry_run_accepts_a_stale_or_closed_market_quotes_sample(tmp_path, monkeypatch):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    _rewrite_quotes(harness, sampled_at_local="2024-05-01T10:00:00", market_state="closed")
    result = driver.compute_and_record(**harness.kwargs(dry_run=True))
    assert "reason_code" not in result


def test_a_catalog_account_type_change_skips_before_any_fetch(tmp_path, monkeypatch):
    """The tax treatment of a one-shot score comes from account_types; a
    reordered/changed catalog value must stop the run, not silently re-tax it."""
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    with open(harness.catalog_path, "r", encoding="utf-8") as handle:
        entries = json.load(handle)
    entries[0]["account_types"] = ["roth_ira", "taxable"]
    with open(harness.catalog_path, "w", encoding="utf-8") as handle:
        json.dump(entries, handle)
    result = driver.compute_and_record(**harness.kwargs())
    assert result["reason_code"] == "parameter_drift"
    assert "account_types" in result["reason"]
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []


def _benchmark_ending(monkeypatch, last_day):
    """Patch the daily benchmark loader to a frame whose LAST row is `last_day`."""
    frame = daily_benchmark_frame(end=last_day)
    monkeypatch.setattr(driver.batch5, "load_frame", lambda symbol, **_kw: (frame, "cache"))
    return frame


def test_a_benchmark_that_ends_before_the_last_intraday_session_refuses_before_the_backtest(
        tmp_path, monkeypatch):
    """batch5.load_frame reuses any cached SPY.csv it finds -- including the one
    a dry run left under the same run-id. A stale benchmark would silently
    drop the newest sessions from the permanent score."""
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    _benchmark_ending(monkeypatch, "2026-03-02")      # intraday bars run to 2026-03-04
    ran = []
    real_backtest = driver.run_real_intraday_backtest
    monkeypatch.setattr(driver, "run_real_intraday_backtest",
                        lambda *a, **k: ran.append(1) or real_backtest(*a, **k))
    with pytest.raises(ValueError, match="benchmark"):
        driver.compute_and_record(**harness.kwargs())
    assert ran == []
    assert harness.ledger_rows() == []


def test_a_benchmark_reaching_the_last_intraday_session_is_accepted(tmp_path, monkeypatch):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    _benchmark_ending(monkeypatch, "2026-03-04")
    result = driver.compute_and_record(**harness.kwargs(dry_run=True))
    assert "reason_code" not in result


def test_real_data_start_is_pinned_to_the_measured_spy_floor():
    """SPY minute bars exist from 2016-01-04 (a full 390-bar session that
    day; earlier dates raise IntradayDataGapError at the vendor). A later start would silently shrink the one-shot
    scored window; an earlier one cannot be fetched."""
    assert driver.REAL_DATA_START == "2016-01-04"


# ====================================== 5. details disclosed on every result

def test_entries_filled_on_the_force_flat_bar_are_counted():
    """RSI oversold at the 15:54 bucket end passes the entry guard (< 15:55),
    fills at the 15:55 open and is force-flattened on that SAME bar: a
    one-minute round trip that pays full cost. The mechanism is silent on it,
    so the driver counts them instead of hiding them."""
    prices = [100.0] * 76 + [99.0, 99.0]      # first down bucket is 15:50-15:54
    trades = driver.run_real_intraday_backtest(_session("2026-03-02", prices), cost_bps=0.0)
    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["entry_datetime"] == row["exit_datetime"] == pd.Timestamp("2026-03-02 15:55", tz=TZ)
    assert driver.count_force_flat_bar_entries(trades) == 1
    normal = driver.run_real_intraday_backtest(_dip_session(), cost_bps=0.0)
    assert driver.count_force_flat_bar_entries(normal) == 0
    assert driver.count_force_flat_bar_entries(normal.iloc[0:0]) == 0


def test_sessions_with_missing_minutes_are_identified_against_the_nyse_calendar():
    """The factory counts observed ROWS, not clock time, to find 5-minute
    bucket ends: after a missing minute every later boundary in that session
    runs a minute late. A half day that is COMPLETE (210 bars) is not flagged."""
    full = _session("2026-11-24", [100.0] * 78)
    gappy = _session("2026-11-25", [100.0] * 78)
    gappy = gappy.drop(gappy.index[100])                       # one minute missing
    half_day = _session("2026-11-27", [100.0] * 42)             # day after Thanksgiving, 13:00 close
    bars = pd.concat([full, gappy, half_day])
    assert driver.incomplete_sessions(bars) == ["2026-11-25"]
    assert driver.incomplete_sessions(pd.concat([full, half_day])) == []


def test_a_dry_run_reports_every_nonzero_detail_in_its_result_and_note(tmp_path, monkeypatch):
    """All three counted details non-zero at once, each from a purpose-built
    session: an early-close half day (2026-11-27, 13:00 close, oversold last
    bucket) whose signal fills at the next session's open; a session whose
    oversold 15:54 bucket end fills on the 15:55 force-flat bar; and a session
    missing one minute."""
    half_day = _session("2026-11-27", [100.0] * 41 + [99.0])          # complete 210-bar half day
    after_half_day = _session("2026-11-30", [99.0] * 78)               # the fill lands here at 09:30
    force_flat_day = _session("2026-12-01", [100.0] * 76 + [99.0, 99.0])
    gappy_day = _session("2026-12-02", [100.0] * 78)
    gappy_day = gappy_day.drop(gappy_day.index[50])
    bars = pd.concat([half_day, after_half_day, force_flat_day, gappy_day])
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    _benchmark_ending(monkeypatch, "2026-12-31")
    monkeypatch.setattr(driver, "_today_et", lambda: "2026-12-31")   # the frames are dated after the harness's "today"
    monkeypatch.setattr(driver.intraday_data, "fetch_intraday_bars",
                        lambda symbol, start_date, end_date, **_kw: bars)
    result = driver.compute_and_record(**harness.kwargs(dry_run=True))
    assert result["entries_filled_at_session_open"] == 1
    assert result["entries_filled_on_force_flat_bar"] == 1
    assert result["n_incomplete_sessions"] == 1
    assert result["incomplete_sessions"] == ["2026-12-02"]
    for fragment in ("entries filled at the session open=1",
                     "entries filled on the force-flat bar=1",
                     "sessions with missing minutes=1"):
        assert fragment in result["cost_model_note"], fragment


def test_a_session_absent_from_the_nyse_calendar_is_flagged_not_a_crash():
    saturday = _session("2026-03-07", [100.0] * 78)               # 2026-03-07 is a Saturday
    assert driver.incomplete_sessions(saturday) == ["2026-03-07"]


def test_the_benchmark_drop_fields_carry_real_values(tmp_path, monkeypatch):
    """A benchmark that starts AFTER the first intraday session must show up as
    dropped sessions and benchmark_truncated=True; a benchmark covering the
    whole span drops at most the one session with no prior close."""
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    covered = driver.compute_and_record(**harness.kwargs(dry_run=True))["window"]
    assert covered["sessions_dropped_to_benchmark"] == 0
    assert covered["benchmark_truncated"] is False

    late = daily_benchmark_frame(start="2026-03-03", end="2026-03-04")
    monkeypatch.setattr(driver.batch5, "load_frame", lambda symbol, **_kw: (late, "cache"))
    truncated = driver.compute_and_record(**harness.kwargs(dry_run=True))["window"]
    assert truncated["sessions_dropped_to_benchmark"] >= 1
    assert truncated["benchmark_truncated"] is True


# ============================ 6. scored branch + signal parameters actually bite

def _scored_real_run_setup(tmp_path, monkeypatch):
    """A harness whose real run reaches a SCORED verdict: ABI-1 needs >= 252
    in-sample and 756 holdout sessions, so ~1,370 real business days (2 bars
    each, except a complete last session) and a synthetic trade set (the
    backtest is stubbed -- the signal has its own tests above) so the SCORED
    branch of compute_and_record, the ledger write and the note run for real."""
    days = pd.bdate_range("2021-01-04", "2026-03-04")
    index = pd.DatetimeIndex([pd.Timestamp(f"{d.date()} {t}", tz=TZ)
                              for d in days for t in ("09:30", "09:31")], name="timestamp")
    index = index[index.date != days[-1].date()]           # every day 2-bar EXCEPT the last...
    px = np.full(len(index), 100.0)
    thin = pd.DataFrame({"Open": px, "High": px, "Low": px, "Close": px, "Volume": 1000.0,
                         "TradeCount": 10, "VWAP": px}, index=index)
    bars = pd.concat([thin, _session(str(days[-1].date()), [100.0] * 78)])   # ...which is complete
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    monkeypatch.setattr(driver.intraday_data, "fetch_intraday_bars",
                        lambda symbol, start_date, end_date, **_kw: bars)
    rng = np.random.default_rng(1)

    def fake_backtest(_bars, *, cost_bps):
        picked = np.sort(rng.choice(len(days), size=900, replace=False))
        return pd.DataFrame([{
            "entry_datetime": pd.Timestamp(f"{days[k].date()} 09:31", tz=TZ),   # not the 09:30 open
            "exit_datetime": pd.Timestamp(f"{days[k].date()} 09:31", tz=TZ),
            "entry_px": 100.0, "exit_px": 100.0, "held_days": 1, "reason": "signal_exit",
            "gross_ret": 0.0, "net_ret": float(rng.normal(0.0002, 0.002))} for k in picked])

    monkeypatch.setattr(driver, "run_real_intraday_backtest", fake_backtest)
    return harness, days


def test_a_real_run_that_reaches_a_scored_verdict_records_it_completely(tmp_path, monkeypatch):
    """Every other driver-level test ends at not_scoreable (3 sessions); this one
    reaches a SCORED verdict (see _scored_real_run_setup)."""
    harness, days = _scored_real_run_setup(tmp_path, monkeypatch)
    result = driver.compute_and_record(**harness.kwargs())

    assert result["stage"] in ("survived", "cut_dsr", "cut_holdout", "cut_risk_refusal")
    assert result["window"]["holdout_sessions"] == 756
    assert isinstance(result["dsr_value"], float)
    outcomes = [r for r in harness.ledger_rows() if r.get("event") == "outcome"]
    assert len(outcomes) == 1 and outcomes[0]["construction"] == "ABI-1"
    assert outcomes[0]["stage"] == result["stage"]
    assert "long side only" in outcomes[0]["note"]
    assert f"attempt={result['attempt_id']}" in outcomes[0]["note"]
    # The three counted events reach the PERMANENT ledger note, not just the result
    # (the stubbed trades have none; every 2-bar test day is short of the
    # scheduled minutes, so every session but the complete last one counts).
    for fragment in ("entries filled at the session open=0",
                     "entries filled on the force-flat bar=0",
                     f"sessions with missing minutes={len(days) - 1}"):
        assert fragment in outcomes[0]["note"], fragment
    json.dumps(driver.batch5._json_safe(result), allow_nan=False)


def test_a_weak_dip_that_stays_above_the_oversold_level_takes_no_entry():
    """Closes 100 x6, 101, 100.5, 99.0 (5-min buckets): Wilder RSI(2) reads
    12.5 from the third move on and never falls below the registered 10."""
    prices = [100.0] * 6 + [101.0, 100.5, 99.0] + [99.0] * (78 - 9)
    closes = [100.0] * 6 + [101.0, 100.5, 99.0]
    assert 10.0 < _wilder_rsi(closes) < 20.0
    assert len(driver.run_real_intraday_backtest(_session("2026-03-02", prices), cost_bps=0.0)) == 0


def test_a_deeper_dip_below_the_oversold_level_takes_the_entry():
    prices = [100.0] * 6 + [101.0, 100.5, 98.0] + [98.0] * (78 - 9)
    closes = [100.0] * 6 + [101.0, 100.5, 98.0]
    assert _wilder_rsi(closes) < 10.0
    trades = driver.run_real_intraday_backtest(_session("2026-03-02", prices), cost_bps=0.0)
    # (On a flat tape RSI stays oversold, so the strategy re-enters after each
    # equality exit: only the FIRST entry is the one this test is about.)
    assert len(trades) >= 1
    assert trades.iloc[0]["entry_datetime"] == pd.Timestamp("2026-03-02 10:15", tz=TZ)


def test_the_exit_uses_the_registered_five_bar_sma_not_a_shorter_one():
    """After the entry the tape bounces to 98.6: that is above the 3-bar SMA
    (98.53) but below the 5-bar SMA (99.12), so the registered 5-bar rule holds
    the position one more bucket and exits at the 10:29 close (99.6)."""
    prices = [100.0] * 8 + [99.0, 98.0, 98.6, 99.6] + [99.6] * (78 - 12)
    trades = driver.run_real_intraday_backtest(_session("2026-03-02", prices), cost_bps=0.0)
    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["entry_datetime"] == pd.Timestamp("2026-03-02 10:15", tz=TZ)
    assert row["exit_datetime"] == pd.Timestamp("2026-03-02 10:29", tz=TZ)
    assert row["exit_px"] == pytest.approx(99.6)


def test_an_rsi_exactly_at_the_oversold_level_takes_no_entry():
    """The mechanism enters when RSI(2) "drops below 10": STRICTLY below.
    Closes 100 x5, 102, 93 give Wilder avg_gain 0.5 / avg_loss 4.5 (exact binary
    fractions), an RSI of exactly 10.0 at the 10:04 bucket end and at every
    later flat bar -- a `<=` implementation would enter there."""
    closes = [100.0] * 5 + [102.0, 93.0]
    assert _wilder_rsi(closes) == 10.0
    prices = closes + [93.0] * (78 - len(closes))
    assert len(driver.run_real_intraday_backtest(_session("2026-03-02", prices), cost_bps=0.0)) == 0


# =============================== 7. clock, cache, attribution

def test_today_et_is_the_new_york_date_not_the_utc_date():
    """01:30 UTC on the 24th is 21:30 ET on the 23rd: a UTC-date implementation
    would wrongly refuse every real run started between 20:00 ET and midnight."""
    assert driver._today_et(now=pd.Timestamp("2026-09-24 01:30", tz="UTC")) == "2026-09-23"
    assert driver._today_et(now=pd.Timestamp("2026-09-24 13:30", tz="UTC")) == "2026-09-24"
    # Sandwich the default-clock read between two reads so a run that crosses
    # New York midnight cannot make this flaky.
    before = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
    got = driver._today_et()
    after = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
    assert got in {before, after}


def test_the_run_reads_the_clock_once_so_the_date_check_and_the_fetch_cutoff_agree(
        tmp_path, monkeypatch):
    """A run straddling midnight ET must not pass the today-rule with one date
    and then cut its fetch off at another."""
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    reads = iter([TESTED_DATE, "2026-09-19", "2026-09-20", "2026-09-21"])
    monkeypatch.setattr(driver, "_today_et", lambda: next(reads))
    cutoffs = []
    monkeypatch.setattr(driver, "_most_recent_available_session",
                        lambda before_date: cutoffs.append(before_date) or "2026-03-04")
    driver.compute_and_record(**harness.kwargs())
    assert cutoffs == [TESTED_DATE]


def test_an_end_date_on_or_after_today_is_refused_even_on_a_dry_run(tmp_path, monkeypatch):
    """Fetching today's still-forming session would write a PARTIAL day into the
    shared minute cache, which fetch_intraday_bars then trusts forever."""
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    with pytest.raises(ValueError, match="end_date"):
        driver.compute_and_record(**harness.kwargs(dry_run=True, end_date=TESTED_DATE))
    assert harness.fetch_calls == []
    result = driver.compute_and_record(**harness.kwargs(dry_run=True, end_date="2026-09-17"))
    assert "reason_code" not in result


def test_benchmark_rows_dated_today_or_later_never_reach_the_scoring(tmp_path, monkeypatch):
    """yfinance's newest daily row is an in-progress bar; whatever else happens,
    a row dated today must never be scored as if final."""
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    _benchmark_ending(monkeypatch, TESTED_DATE)          # includes a row dated TODAY
    seen = {}
    real_evaluate = driver.abi1_driver.evaluate_family

    def spy_evaluate(curve, **kwargs):
        seen["benchmark_last"] = kwargs["benchmark_frame"].index.max()
        return real_evaluate(curve, **kwargs)

    monkeypatch.setattr(driver.abi1_driver, "evaluate_family", spy_evaluate)
    driver.compute_and_record(**harness.kwargs(dry_run=True))
    assert seen["benchmark_last"] < pd.Timestamp(TESTED_DATE)


def test_a_stale_benchmark_refusal_deletes_the_cache_it_read_so_a_retry_refetches(tmp_path, monkeypatch):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    frame = daily_benchmark_frame(end="2026-03-02")
    cache_dir = os.path.join(harness.research_root, "guard-test-run", "artifacts", f"ohlc-{TESTED_DATE}")
    os.makedirs(cache_dir)
    cached = os.path.join(cache_dir, "SPY.csv")
    with open(cached, "w", encoding="utf-8") as handle:
        handle.write("stale")
    monkeypatch.setattr(driver.batch5, "load_frame", lambda symbol, **_kw: (frame, "cache"))
    with pytest.raises(ValueError, match="benchmark"):
        driver.compute_and_record(**harness.kwargs())
    assert not os.path.exists(cached)


def test_the_cache_directory_is_scoped_to_the_run_id(tmp_path, monkeypatch):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    dirs = []
    frame = daily_benchmark_frame()
    monkeypatch.setattr(driver.batch5, "load_frame",
                        lambda symbol, *, ohlc_loader, cache_dir, **_kw: dirs.append(cache_dir) or (frame, "x"))
    driver.compute_and_record(**harness.kwargs(dry_run=True, run_id="my-unique-run"))
    assert "my-unique-run" in dirs[0]


@pytest.mark.parametrize("market_state", [
    "closed (opens 09:30 ET Monday)",
    "open (pre-market)",
    "open (after-hours session)",
    "extended hours",
])
def test_a_real_run_refuses_a_non_regular_or_closed_market_state(tmp_path, monkeypatch, market_state):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    _rewrite_quotes(harness, market_state=market_state)
    with pytest.raises(ValueError, match="market_state"):
        driver.compute_and_record(**harness.kwargs())
    assert harness.fetch_calls == []


def test_a_real_run_accepts_the_intraday_capture_market_state(tmp_path, monkeypatch):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    _rewrite_quotes(harness, market_state="open (intraday capture)")
    assert "reason_code" not in driver.compute_and_record(**harness.kwargs())


def test_the_result_and_note_report_the_last_intraday_bar(tmp_path, monkeypatch):
    """A partial (still-forming or cache-truncated) final session shows up as a
    last bar well before the close; a human reading the note can see it."""
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    result = driver.compute_and_record(**harness.kwargs(dry_run=True))
    assert result["last_intraday_bar"] == "2026-03-04 15:59"      # the harness's complete final session
    assert "last intraday bar=2026-03-04 15:59" in result["cost_model_note"]


def test_the_ledger_note_attributes_the_code_that_produced_the_score(tmp_path, monkeypatch):
    """record_abi1_outcome ignores signal_source_* for SCORED stages, so the
    driver and signal-module identities go into the note itself."""
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    result = driver.compute_and_record(
        **harness.kwargs(dry_run=True, signal_source_sha256="abcdef" * 10 + "abcd"))
    note = result["cost_model_note"]
    assert "code: driver=scripts/test_driver.py@abcdefabcdef" in note
    assert "day_trading_signals.py@" in note
    import mining.evidence as evidence_mod
    assert evidence_mod.source_sha256(driver.mod.__file__)[:12] in note


def test_main_prints_the_result_and_exits_nonzero_if_the_results_file_cannot_be_written(
        tmp_path, monkeypatch, capsys):
    """The ledger row is written BEFORE main() writes the results file. If that
    write fails, the row is permanent but its detail artifact is missing -- so
    the full result must already be on stdout and the exit code non-zero."""
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)

    def broken_write(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(driver.batch3, "_atomic_write_text", broken_write)
    exit_code = driver.main(harness.cli_argv())
    assert exit_code == 1
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["output"] is None
    assert "disk full" in printed["write_error"]
    assert printed["result"]["family_id"] == driver.FAMILY_ID
    outcomes = [r for r in harness.ledger_rows() if r.get("event") == "outcome"]
    assert len(outcomes) == 1        # the permanent row exists; the printout is its detail


def test_a_benchmark_left_empty_after_dropping_todays_row_is_refused(tmp_path, monkeypatch):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    only_today = daily_benchmark_frame(start=TESTED_DATE, end=TESTED_DATE)
    monkeypatch.setattr(driver.batch5, "load_frame", lambda symbol, **_kw: (only_today, "cache"))
    with pytest.raises(ValueError, match="benchmark frame is empty"):
        driver.compute_and_record(**harness.kwargs(dry_run=True))


def test_a_real_run_refuses_when_the_last_intraday_session_is_incomplete(tmp_path, monkeypatch):
    """The minute cache trusts any existing day file, so a session cached while
    still forming is a PARTIAL day. The newest session is the one most exposed
    to that, and it is scored into a permanent row: refuse a real run whose last
    session is short of its scheduled minutes (a dry run only reports it)."""
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    partial = minute_bar_frame()                     # last session: 2 bars only
    monkeypatch.setattr(driver.intraday_data, "fetch_intraday_bars",
                        lambda symbol, start_date, end_date, **_kw: partial)
    ran = []
    real_backtest = driver.run_real_intraday_backtest
    monkeypatch.setattr(driver, "run_real_intraday_backtest",
                        lambda *a, **k: ran.append(1) or real_backtest(*a, **k))
    with pytest.raises(ValueError, match="last intraday session"):
        driver.compute_and_record(**harness.kwargs())
    assert ran == [] and harness.ledger_rows() == []
    dry = driver.compute_and_record(**harness.kwargs(dry_run=True))
    assert "reason_code" not in dry and dry["incomplete_sessions"][-1] == "2026-03-04"


def test_a_failure_after_the_outcome_row_is_written_still_surfaces_the_result(
        tmp_path, monkeypatch, capsys):
    """record_abi1_outcome appends the permanent OUTCOME row first and registers
    the first_test row second. If that second write fails the outcome is
    already permanent: the result must not be lost with the traceback."""
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)

    def broken_first_test(*_args, **_kwargs):
        raise RuntimeError("ledger lock timeout")

    monkeypatch.setattr(driver.trial_ledger, "record_first_test", broken_first_test)
    result = driver.compute_and_record(**harness.kwargs())
    assert "lock timeout" in result["ledger_post_write_error"]
    assert result["family_id"] == driver.FAMILY_ID and "verdict" in result
    assert result["stage"] == "insufficient_evidence"        # 3 sessions: not scoreable
    assert len([r for r in harness.ledger_rows() if r.get("event") == "outcome"]) == 1


def test_main_exits_nonzero_but_prints_and_writes_when_the_first_test_registration_fails(
        tmp_path, monkeypatch, capsys):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    monkeypatch.setattr(driver.trial_ledger, "record_first_test",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ledger lock timeout")))
    assert driver.main(harness.cli_argv()) == 1
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert "lock timeout" in printed["result"]["ledger_post_write_error"]
    assert os.path.exists(os.path.join(harness.results_dir(), "intraday-rsi2-reversion-results.json"))


def test_a_failure_before_any_outcome_row_exists_propagates_and_spends_nothing(tmp_path, monkeypatch):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)

    def broken_record(*_args, **_kwargs):
        raise RuntimeError("cannot open ledger")

    monkeypatch.setattr(driver.trial_ledger, "record_outcome_if_not_scored", broken_record)
    with pytest.raises(RuntimeError, match="cannot open ledger"):
        driver.compute_and_record(**harness.kwargs())
    assert harness.ledger_rows() == []


def test_a_post_write_failure_on_a_scored_run_reports_the_scored_stage(tmp_path, monkeypatch):
    harness, _days = _scored_real_run_setup(tmp_path, monkeypatch)

    def broken_first_test(*_args, **_kwargs):
        raise RuntimeError("ledger lock timeout")

    monkeypatch.setattr(driver.trial_ledger, "record_first_test", broken_first_test)
    result = driver.compute_and_record(**harness.kwargs())
    outcomes = [r for r in harness.ledger_rows() if r.get("event") == "outcome"]
    assert len(outcomes) == 1
    assert result["stage"] in ("survived", "cut_dsr", "cut_holdout", "cut_risk_refusal")
    assert result["stage"] == outcomes[0]["stage"]
    assert "lock timeout" in result["ledger_post_write_error"]


def _seed_other_runs_abi1_outcome(harness):
    from mining import trial_ledger
    trial_ledger.record_outcome(
        harness.ledger_path, driver.FAMILY_ID, "OTHER-RUN", "2026-09-18",
        stage="cut_dsr", dsr_value=0.01, sharpe_in_sample=0.1, construction="ABI-1",
        note="the concurrent winner")


def test_a_concurrent_winner_is_not_reported_as_this_runs_post_write_failure(tmp_path, monkeypatch):
    """This run's own write fails BEFORE appending, and another process records
    the family's ABI-1 outcome in the meantime. The permanent row is the
    other run's: this run's result must be discarded, not returned as if it
    were the recorded one."""
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)

    def racing_record(*_args, **_kwargs):
        _seed_other_runs_abi1_outcome(harness)
        raise RuntimeError("boom")

    monkeypatch.setattr(driver.trial_ledger, "record_outcome_if_not_scored", racing_record)
    result = driver.compute_and_record(**harness.kwargs())
    assert result["reason_code"] == "already_scored_abi1"
    outcomes = [r for r in harness.ledger_rows() if r.get("event") == "outcome"]
    assert [r["run_id"] for r in outcomes] == ["OTHER-RUN"]


def test_an_unreadable_ledger_after_a_failed_write_still_surfaces_the_result(tmp_path, monkeypatch):
    """A torn ledger line is the likely reason the second write failed, and it
    also makes read_all raise inside the handler: the result must survive with
    an explicit 'ledger state unknown' flag instead of the traceback."""
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    real_read_all = driver.trial_ledger.read_all
    failed = []

    def failing_record(*_args, **_kwargs):
        failed.append(True)
        raise RuntimeError("first write failed")

    def read_all_after_failure(path):
        if failed:
            raise ValueError("torn ledger line")
        return real_read_all(path)

    monkeypatch.setattr(driver.trial_ledger, "record_outcome_if_not_scored", failing_record)
    monkeypatch.setattr(driver.trial_ledger, "read_all", read_all_after_failure)
    result = driver.compute_and_record(**harness.kwargs())
    assert "ledger state unknown" in result["ledger_post_write_error"]
    assert "first write failed" in result["ledger_post_write_error"]
    assert "torn ledger line" in result["ledger_post_write_error"]


def test_a_legacy_per_trade_row_does_not_make_a_pre_write_failure_look_written(tmp_path, monkeypatch):
    """With a legacy per_trade outcome already in the ledger, the
    post-failure re-read must key on construction == ABI-1, or it would swallow
    every pre-write failure as if the slot had been spent."""
    from mining import trial_ledger
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    trial_ledger.record_outcome(
        harness.ledger_path, driver.FAMILY_ID, "legacy-run", "2020-01-02",
        stage="cut_dsr", dsr_value=0.01, sharpe_in_sample=0.1)          # construction=per_trade

    def broken_record(*_args, **_kwargs):
        raise RuntimeError("cannot open ledger")

    monkeypatch.setattr(trial_ledger, "record_outcome_if_not_scored", broken_record)
    with pytest.raises(RuntimeError, match="cannot open ledger"):
        driver.compute_and_record(**harness.kwargs())


def test_another_familys_abi1_row_does_not_make_a_pre_write_failure_look_written(tmp_path, monkeypatch):
    from mining import trial_ledger
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    trial_ledger.record_outcome(
        harness.ledger_path, "some_other_family", "other-run", "2026-09-18",
        stage="cut_dsr", dsr_value=0.01, sharpe_in_sample=0.1, construction="ABI-1")

    def broken_record(*_args, **_kwargs):
        raise RuntimeError("cannot open ledger")

    monkeypatch.setattr(trial_ledger, "record_outcome_if_not_scored", broken_record)
    with pytest.raises(RuntimeError, match="cannot open ledger"):
        driver.compute_and_record(**harness.kwargs())


def test_a_session_with_a_missing_minute_and_a_duplicate_row_is_still_incomplete():
    """Row COUNT alone is evadable: one minute missing plus one duplicated
    timestamp is still 390 rows. Completeness is the exact scheduled minute grid."""
    day = _session("2026-11-24", [100.0] * 78)
    missing_one_plus_duplicate = pd.concat([day.drop(day.index[100]), day.iloc[[50]]]).sort_index()
    assert len(missing_one_plus_duplicate) == 390
    assert driver.incomplete_sessions(missing_one_plus_duplicate) == ["2026-11-24"]
    assert driver.incomplete_sessions(day) == []


def test_a_winner_sharing_this_runs_run_id_and_date_is_still_not_mistaken_for_its_own_row(
        tmp_path, monkeypatch):
    """Two invocations with the default run id on the same day share run_id and
    tested_date. Only a per-attempt token persisted in the outcome row can tell
    whose row it is."""
    from mining import trial_ledger
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)

    def racing_record(_ledger, family_id, run_id, tested_date, **_kwargs):
        trial_ledger.record_outcome(
            harness.ledger_path, family_id, run_id, tested_date,      # SAME run_id and date
            stage="cut_dsr", dsr_value=0.01, sharpe_in_sample=0.1, construction="ABI-1",
            note="the concurrent winner")
        raise RuntimeError("boom")

    monkeypatch.setattr(driver.trial_ledger, "record_outcome_if_not_scored", racing_record)
    result = driver.compute_and_record(**harness.kwargs())
    assert result["reason_code"] == "already_scored_abi1"


def test_every_attempt_carries_a_unique_token_that_reaches_the_ledger_note(tmp_path, monkeypatch):
    harness = RsiHarness(tmp_path, monkeypatch, driver,
                         quote_symbols=QUOTE_SYMBOLS, catalog_tickers=TICKERS)
    first = driver.compute_and_record(**harness.kwargs(dry_run=True))
    second = driver.compute_and_record(**harness.kwargs(dry_run=True))
    assert first["attempt_id"] != second["attempt_id"]
    assert f"attempt={first['attempt_id']}" in first["cost_model_note"]
    real = driver.compute_and_record(**harness.kwargs())
    outcome = [r for r in harness.ledger_rows() if r.get("event") == "outcome"][0]
    # (this 3-session run is UNSCORED: the row has no note, only evidence)
    assert f"attempt={real['attempt_id']}" in json.dumps(outcome)
