"""Driver-level tests for scripts/run_green_day_morning_pullback_abi1.py --
the bespoke ABI-1 driver for green_day_morning_pullback_continuation_
daytrade, pooled SPY/QQQ.

Reuses the SAME hermetic harness (`_intraday_abi1_driver_support.Harness`)
the three sibling intraday drivers' shared guard-test file already relies
on, but as its OWN dedicated file rather than folding into
`test_intraday_abi1_driver_guards.py`'s parametrized matrix. Every guard
that matrix exercises is covered here too, just via a dedicated
(unparametrized) copy.

Every test is hermetic: the ledger, catalog, quotes file and research root
all live under `tmp_path` and the network is monkeypatched. The REAL trial
ledger is never touched.
"""
import json
import os
import sys
import threading
import time

import pandas as pd
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
for _path in (_SRC, _SCRIPTS, os.path.dirname(__file__)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_green_day_morning_pullback_abi1 as driver  # noqa: E402
from _intraday_abi1_driver_support import Harness, write_quotes  # noqa: E402

QUOTE_SYMBOLS = ("SPY", "QQQ")
POOL_TICKERS = ["SPY", "QQQ"]


@pytest.fixture
def harness(tmp_path, monkeypatch):
    return Harness(tmp_path, monkeypatch, driver,
                   quote_symbols=QUOTE_SYMBOLS, catalog_tickers=POOL_TICKERS)


# ------------------------------------------------ real-run window overrides

def test_real_run_refuses_an_end_date_override(harness):
    with pytest.raises(ValueError, match="end_date"):
        driver.compute_and_record(**harness.kwargs(end_date="2026-03-03"))
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []


def test_dry_run_still_accepts_an_end_date_override(harness):
    result = driver.compute_and_record(**harness.kwargs(dry_run=True, end_date="2026-03-03"))
    assert "reason_code" not in result
    assert harness.fetch_calls
    assert all(end == "2026-03-03" for _symbol, _start, end in harness.fetch_calls)
    assert harness.ledger_rows() == []


def test_real_run_refuses_a_start_date_override(harness):
    with pytest.raises(ValueError, match="start_date"):
        driver.compute_and_record(**harness.kwargs(start_date="2026-03-03"))
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []


def test_dry_run_still_accepts_a_start_date_override(harness):
    result = driver.compute_and_record(**harness.kwargs(dry_run=True, start_date="2026-03-02"))
    assert "reason_code" not in result
    assert all(start == "2026-03-02" for _symbol, start, _end in harness.fetch_calls)


def test_a_real_run_always_fetches_from_the_measured_floor(harness):
    driver.compute_and_record(**harness.kwargs())
    assert harness.fetch_calls
    assert all(start == driver.REAL_DATA_START for _symbol, start, _end in harness.fetch_calls)


# ----------------------------------------------------------- quotes guard

def test_real_run_refuses_a_synthetic_quotes_file(harness):
    write_quotes(harness.quotes_path, QUOTE_SYMBOLS, synthetic=True)
    with pytest.raises(ValueError, match="SYNTHETIC"):
        driver.compute_and_record(**harness.kwargs())
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []


# ------------------------------------------------- ledger side effects

def test_a_dry_run_writes_no_ledger_rows(harness):
    write_quotes(harness.quotes_path, QUOTE_SYMBOLS, synthetic=True)
    result = driver.compute_and_record(**harness.kwargs(dry_run=True))
    assert "reason_code" not in result
    assert harness.fetch_calls
    assert "verdict" in result and "stage" in result
    assert harness.ledger_rows() == []
    assert not os.path.exists(harness.ledger_path)


def test_a_real_run_writes_exactly_one_abi1_outcome_and_a_second_is_blocked(harness):
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


def test_two_concurrent_real_runs_only_one_outcome_is_ever_recorded(harness, monkeypatch):
    """Same race the concurrency guard closes for the three
    sibling drivers (see trial_ledger.record_outcome_if_not_scored):
    compute_and_record's own already_scored prefilter (an unlocked read,
    run BEFORE the real backtest/evaluate work) and the eventual outcome
    append are two separate operations with real compute in between.
    Genuine threads, real compute through the real backtest+evaluate+
    ledger-write path -- not a mock."""
    from mining import trial_ledger

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


def test_a_seeded_abi1_outcome_blocks_a_real_run_before_any_fetch(harness):
    harness.seed_abi1_outcome()
    seeded = harness.ledger_rows()
    result = driver.compute_and_record(**harness.kwargs())
    assert result["reason_code"] == "already_scored_abi1"
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == seeded


# ----------------------------------------------------- catalog/registration

def test_a_non_eligible_catalog_status_skips_before_any_fetch(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch, driver, quote_symbols=QUOTE_SYMBOLS,
                      catalog_tickers=POOL_TICKERS, status="ruled_out")
    result = driver.compute_and_record(**harness.kwargs())
    assert result["reason_code"] == "catalog_status"
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []


def test_a_catalog_parameter_drift_skips_before_any_fetch(tmp_path, monkeypatch):
    drifted = dict(driver.PARAMETERS)
    drifted["force_flat_et"] = "15:30"
    harness = Harness(tmp_path, monkeypatch, driver, quote_symbols=QUOTE_SYMBOLS,
                      catalog_tickers=POOL_TICKERS, catalog_parameters=drifted)
    result = driver.compute_and_record(**harness.kwargs())
    assert result["reason_code"] == "parameter_drift"
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []


def test_a_catalog_ticker_drift_skips_before_any_fetch(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch, driver, quote_symbols=QUOTE_SYMBOLS,
                      catalog_tickers=["SPY", "IWM"])
    result = driver.compute_and_record(**harness.kwargs())
    assert result["reason_code"] == "parameter_drift"
    assert "tickers" in result["reason"]
    assert harness.fetch_calls == []
    assert harness.ledger_rows() == []


def test_a_catalog_ticker_order_difference_is_not_drift(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch, driver, quote_symbols=QUOTE_SYMBOLS,
                      catalog_tickers=list(reversed(POOL_TICKERS)))
    result = driver.compute_and_record(**harness.kwargs(dry_run=True))
    assert "reason_code" not in result


# ------------------------------------------------------------ main() writes

def test_main_refuses_a_real_run_end_date_before_writing_anything(harness):
    with pytest.raises(ValueError, match="end_date"):
        driver.main(harness.cli_argv("--end-date", "2026-03-03"))
    assert not os.path.exists(harness.results_dir())
    assert harness.ledger_rows() == []


def test_main_writes_the_results_file_for_a_real_run(harness, capsys):
    assert driver.main(harness.cli_argv()) == 0
    path = os.path.join(harness.results_dir(), "green-day-morning-pullback-pooled-results.json")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["dry_run"] is False
    assert payload["result"]["family_id"] == driver.FAMILY_ID
    assert "verdict" in payload["result"]


def test_main_never_overwrites_the_real_results_file_with_a_skip(harness, capsys):
    assert driver.main(harness.cli_argv()) == 0
    path = os.path.join(harness.results_dir(), "green-day-morning-pullback-pooled-results.json")
    with open(path, "rb") as handle:
        first_bytes = handle.read()
    capsys.readouterr()

    assert driver.main(harness.cli_argv()) == 0
    with open(path, "rb") as handle:
        assert handle.read() == first_bytes
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["result"]["reason_code"] == "already_scored_abi1"
    assert printed["output"] is None


def test_main_writes_no_results_file_when_the_result_is_a_skip(harness, capsys):
    harness.seed_abi1_outcome()
    assert driver.main(harness.cli_argv()) == 0
    assert not os.path.exists(
        os.path.join(harness.results_dir(), "green-day-morning-pullback-pooled-results.json"))
    assert not os.path.exists(
        os.path.join(harness.results_dir(), "green-day-morning-pullback-pooled-dry-run.json"))


def test_main_writes_the_dry_run_file_for_a_dry_run(harness, capsys):
    write_quotes(harness.quotes_path, QUOTE_SYMBOLS, synthetic=True)
    assert driver.main(harness.cli_argv("--dry-run")) == 0
    assert os.path.exists(
        os.path.join(harness.results_dir(), "green-day-morning-pullback-pooled-dry-run.json"))
    assert not os.path.exists(
        os.path.join(harness.results_dir(), "green-day-morning-pullback-pooled-results.json"))
    assert harness.ledger_rows() == []


# ------------------------------------------------------ signal isolation

def test_fresh_signal_fn_per_ticker_no_leaked_state(tmp_path, monkeypatch):
    """A real rally/pause/breakout shape for SPY and a flat no-rally shape
    for QQQ, fetched through the SAME driver invocation -- if the two
    tickers' closures ever shared state (e.g. the same signal_fn instance,
    or state leaking through a module-level variable), QQQ's flat frame
    would either wrongly inherit SPY's in-progress setup or corrupt SPY's
    own trade count. Asserts the per-ticker trade split is exactly what
    each ticker's OWN data independently implies.

    Uses the driver's REAL (catalog) parameters -- min_consolidation_bars=15
    -- not the smaller overrides test_day_trading_green_day_pullback.py
    uses for tractability, so SPY's fixture needs a real 15-bar pause."""
    TZ = "America/New_York"

    def _ts(date_str, time_str):
        return pd.Timestamp(f"{date_str} {time_str}", tz=TZ)

    def _row(o, h, l, c, v=1000.0):
        return {"Open": float(o), "High": float(h), "Low": float(l), "Close": float(c),
                "Volume": float(v), "TradeCount": 10, "VWAP": float(c)}

    def _frame(rows):
        index = pd.DatetimeIndex([ts for ts, _ in rows])
        cols = ["Open", "High", "Low", "Close", "Volume", "TradeCount", "VWAP"]
        data = {col: [r[col] for _, r in rows] for col in cols}
        return pd.DataFrame(data, index=index).sort_index()

    day = "2026-03-04"
    spy_row_list = [
        (_ts(day, "09:30"), _row(100.00, 100.00, 100.00, 100.00)),
        (_ts(day, "09:31"), _row(100.00, 100.20, 100.00, 100.20)),  # +0.20% >= 0.15% rally
    ]
    # 15 identical shallow-pause bars, 09:32-09:46 inclusive (bars_since_new_
    # high reaches 15 on the 09:46 bar; retrace 0.08 of the 0.20 leg, well
    # under the 50% ceiling; 09:46 falls inside the 09:45-11:30 window).
    for minute in range(32, 47):
        spy_row_list.append(
            (_ts(day, f"09:{minute:02d}"), _row(100.15, 100.18, 100.12, 100.15)))
    spy_row_list += [
        (_ts(day, "09:47"), _row(100.15, 100.15, 100.13, 100.25)),  # breakout close > 100.20
        (_ts(day, "09:48"), _row(100.26, 100.30, 100.24, 100.28)),  # fill bar; last bar of the
        # frame -- force-closed "end_of_data" here regardless of stop/target,
        # so this fixture doesn't need to hand-hit either.
    ]
    spy_rows = _frame(spy_row_list)
    qqq_rows = _frame([
        (_ts(day, "09:30"), _row(200.00, 200.05, 199.95, 200.00)),
        (_ts(day, "09:31"), _row(200.00, 200.10, 199.90, 200.02)),  # +0.05% < 0.15%, no rally
        (_ts(day, "09:32"), _row(200.02, 200.08, 199.92, 199.98)),
    ])
    bars_by_ticker = {"SPY": spy_rows, "QQQ": qqq_rows}

    harness = Harness(tmp_path, monkeypatch, driver,
                      quote_symbols=QUOTE_SYMBOLS, catalog_tickers=POOL_TICKERS)

    def fake_fetch(symbol, start_date, end_date, **_kwargs):
        harness.fetch_calls.append((symbol, start_date, end_date))
        return bars_by_ticker[symbol]

    monkeypatch.setattr(driver.intraday_data, "fetch_intraday_bars", fake_fetch)

    result = driver.compute_and_record(**harness.kwargs(dry_run=True))
    assert "reason_code" not in result
    assert result["n_trades_by_ticker"] == {"SPY": 1, "QQQ": 0}
    assert result["n_trades"] == 1
