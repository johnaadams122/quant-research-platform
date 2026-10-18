import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
SCRIPTS_ROOT = os.path.join(REPO_ROOT, "scripts")
for _path in (SRC_ROOT, SCRIPTS_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_bitcoin_spillover_abi1 as driver


# --------------------------------------------------------------- fixtures

def _ohlc_frame(start, end, *, seed, drift):
    index = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(drift, 0.01, len(index)))
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close,
         "Volume": 1_000_000}, index=index)


# Deliberately DIFFERENT seeds/drift for QQQ vs SPY: identical fixtures would
# give zero test power to catch a QQQ/SPY mixup (the same convention as
# test_run_fed_net_liquidity_qqq_mining.py).
def _qqq_frame(start="2015-01-02", end="2020-12-30"):
    return _ohlc_frame(start, end, seed=11, drift=0.0005)


def _spy_frame(start="2015-01-02", end="2020-12-30"):
    return _ohlc_frame(start, end, seed=55, drift=0.0001)


def _btc_close(start="2015-01-02", end="2020-12-30", *, seed=33, drift=0.0002):
    # BTC trades every CALENDAR day (unlike QQQ/SPY's business-day
    # calendar) -- matches the module's own "BTC's own calendar" contract.
    index = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(seed)
    close = 20_000.0 * np.cumprod(1.0 + rng.normal(drift, 0.03, len(index)))
    series = pd.Series(close, index=index, name="close")
    series.attrs.update({
        "source_url": "https://api.kraken.com/0/public/OHLC",
        "symbol": "BTC-USD", "kraken_symbol": "XBTUSD", "interval": 1440,
        "last_row_closed": True, "retrieved_at": "2026-09-17T00:00:00Z",
        "raw_sha256": "test-fixture-sha256",
    })
    return series


def _write_fixture(tmp_path, *, status="eligible", parameters=None):
    ledger_path = str(tmp_path / "trial_ledger.jsonl")
    catalog_path = str(tmp_path / "catalog.json")
    accepted_path = str(tmp_path / "accepted.json")
    qqq_quotes_path = str(tmp_path / "qqq_quotes.json")
    spy_quotes_path = str(tmp_path / "spy_quotes.json")

    catalog_parameters = parameters if parameters is not None else driver.btc_mod.PARAMETERS
    catalog_entry = {
        "family_id": driver.btc_mod.FAMILY_ID, "name": "test",
        "mechanism": "test", "tickers": ["QQQ"], "status": status,
        "account_types": ["taxable", "roth"], "source": "test",
        "added_date": "2026-09-17",
        "parameters": catalog_parameters,
    }
    with open(catalog_path, "w", encoding="utf-8") as handle:
        json.dump([catalog_entry], handle)
    accepted_entry = {
        "family_id": driver.btc_mod.FAMILY_ID, "shape": "single_long",
        "tickers": ["QQQ"], "parameters": driver.btc_mod.PARAMETERS,
    }
    with open(accepted_path, "w", encoding="utf-8") as handle:
        json.dump({"accepted": [accepted_entry]}, handle)
    with open(qqq_quotes_path, "w", encoding="utf-8") as handle:
        json.dump({"sampled_at_local": "2026-09-17T22:00:00",
                   "market_state": "closed", "quotes": {"QQQ": {
            "bid": 600.0, "ask": 600.60, "spread_bps": 10.0,
            "security_status": "Closed", "suspect": False}}}, handle)
    with open(spy_quotes_path, "w", encoding="utf-8") as handle:
        json.dump({"sampled_at_local": "2026-09-17T10:00:00",
                   "market_state": "open", "quotes": {"SPY": {
            "bid": 700.0, "ask": 700.14, "spread_bps": 2.0,
            "security_status": "Normal", "suspect": False}}}, handle)
    return ledger_path, catalog_path, accepted_path, qqq_quotes_path, spy_quotes_path


def _default_kwargs(**overrides):
    kwargs = dict(
        qqq_frame=_qqq_frame(), spy_frame=_spy_frame(), btc_close=_btc_close(),
        run_id="btc-spillover-test", tested_date="2026-09-17", dry_run=False,
        signal_source_path="scripts/run_bitcoin_spillover_abi1.py",
        signal_source_sha256="deadbeef")
    kwargs.update(overrides)
    return kwargs


# ------------------------------------------------------ compute_and_record

def test_compute_and_record_real_run_writes_one_ledger_row(tmp_path):
    (ledger_path, catalog_path, accepted_path,
     qqq_quotes_path, spy_quotes_path) = _write_fixture(tmp_path)

    result = driver.compute_and_record(**_default_kwargs(
        catalog_path=catalog_path, accepted_path=accepted_path,
        qqq_quotes_path=qqq_quotes_path, spy_quotes_path=spy_quotes_path,
        ledger_path=ledger_path))
    assert "reason_code" not in result
    assert result["benchmark"] == "SPY"
    assert result["tickers"] == ["QQQ"]

    with open(ledger_path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    outcomes = [r for r in rows if r.get("event") == "outcome"]
    assert len(outcomes) == 1
    assert outcomes[0]["family_id"] == driver.btc_mod.FAMILY_ID


def test_compute_and_record_skips_when_catalog_status_not_eligible(tmp_path):
    (ledger_path, catalog_path, accepted_path,
     qqq_quotes_path, spy_quotes_path) = _write_fixture(tmp_path, status="ruled_out")

    result = driver.compute_and_record(**_default_kwargs(
        catalog_path=catalog_path, accepted_path=accepted_path,
        qqq_quotes_path=qqq_quotes_path, spy_quotes_path=spy_quotes_path,
        ledger_path=ledger_path, dry_run=True))
    assert result["reason_code"] == "catalog_status"


def test_compute_and_record_skips_a_second_real_run_as_already_scored(tmp_path):
    (ledger_path, catalog_path, accepted_path,
     qqq_quotes_path, spy_quotes_path) = _write_fixture(tmp_path)

    first = driver.compute_and_record(**_default_kwargs(
        catalog_path=catalog_path, accepted_path=accepted_path,
        qqq_quotes_path=qqq_quotes_path, spy_quotes_path=spy_quotes_path,
        ledger_path=ledger_path, run_id="btc-spillover-test-1"))
    assert "reason_code" not in first

    second = driver.compute_and_record(**_default_kwargs(
        catalog_path=catalog_path, accepted_path=accepted_path,
        qqq_quotes_path=qqq_quotes_path, spy_quotes_path=spy_quotes_path,
        ledger_path=ledger_path, run_id="btc-spillover-test-2"))
    assert second["reason_code"] == "already_scored_abi1"


def test_compute_and_record_skips_on_parameter_drift(tmp_path):
    drifted = dict(driver.btc_mod.PARAMETERS)
    drifted["btc_zscore_riskoff_threshold"] = -3.0
    (ledger_path, catalog_path, accepted_path,
     qqq_quotes_path, spy_quotes_path) = _write_fixture(tmp_path, parameters=drifted)

    result = driver.compute_and_record(**_default_kwargs(
        catalog_path=catalog_path, accepted_path=accepted_path,
        qqq_quotes_path=qqq_quotes_path, spy_quotes_path=spy_quotes_path,
        ledger_path=ledger_path, dry_run=True))
    assert result["reason_code"] == "parameter_drift"


def test_compute_and_record_skips_when_not_in_accepted(tmp_path):
    (ledger_path, catalog_path, accepted_path,
     qqq_quotes_path, spy_quotes_path) = _write_fixture(tmp_path)
    with open(accepted_path, "w", encoding="utf-8") as handle:
        json.dump({"accepted": []}, handle)

    result = driver.compute_and_record(**_default_kwargs(
        catalog_path=catalog_path, accepted_path=accepted_path,
        qqq_quotes_path=qqq_quotes_path, spy_quotes_path=spy_quotes_path,
        ledger_path=ledger_path, dry_run=True))
    assert result["reason_code"] == "not_in_accepted"


def _alt_spy_frame():
    # Materially different price path (much higher drift) -- isolates
    # spy_frame's own causal effect on the recorded result (same
    # mixup-detection convention as test_run_fed_net_liquidity_qqq_mining.py).
    return _ohlc_frame("2015-01-02", "2020-12-30", seed=55, drift=0.0090)


def _alt_qqq_frame():
    return _ohlc_frame("2015-01-02", "2020-12-30", seed=11, drift=0.0090)


def test_varying_spy_frame_alone_changes_the_recorded_result(tmp_path):
    (ledger_path, catalog_path, accepted_path,
     qqq_quotes_path, spy_quotes_path) = _write_fixture(tmp_path)
    fixed_qqq_frame = _qqq_frame()
    fixed_btc_close = _btc_close()

    with_spy_a = driver.compute_and_record(**_default_kwargs(
        qqq_frame=fixed_qqq_frame, spy_frame=_spy_frame(), btc_close=fixed_btc_close,
        catalog_path=catalog_path, accepted_path=accepted_path,
        qqq_quotes_path=qqq_quotes_path, spy_quotes_path=spy_quotes_path,
        ledger_path=ledger_path, run_id="btc-spillover-test-a", dry_run=True))

    with_spy_b = driver.compute_and_record(**_default_kwargs(
        qqq_frame=fixed_qqq_frame, spy_frame=_alt_spy_frame(), btc_close=fixed_btc_close,
        catalog_path=catalog_path, accepted_path=accepted_path,
        qqq_quotes_path=qqq_quotes_path, spy_quotes_path=spy_quotes_path,
        ledger_path=ledger_path, run_id="btc-spillover-test-b", dry_run=True))

    assert with_spy_a["sharpe_daily_excess"] != pytest.approx(with_spy_b["sharpe_daily_excess"])
    assert with_spy_a["dsr_value"] != pytest.approx(with_spy_b["dsr_value"])


def test_varying_qqq_frame_alone_changes_the_recorded_result(tmp_path):
    (ledger_path, catalog_path, accepted_path,
     qqq_quotes_path, spy_quotes_path) = _write_fixture(tmp_path)
    fixed_spy_frame = _spy_frame()
    fixed_btc_close = _btc_close()

    with_qqq_a = driver.compute_and_record(**_default_kwargs(
        qqq_frame=_qqq_frame(), spy_frame=fixed_spy_frame, btc_close=fixed_btc_close,
        catalog_path=catalog_path, accepted_path=accepted_path,
        qqq_quotes_path=qqq_quotes_path, spy_quotes_path=spy_quotes_path,
        ledger_path=ledger_path, run_id="btc-spillover-test-c", dry_run=True))

    with_qqq_b = driver.compute_and_record(**_default_kwargs(
        qqq_frame=_alt_qqq_frame(), spy_frame=fixed_spy_frame, btc_close=fixed_btc_close,
        catalog_path=catalog_path, accepted_path=accepted_path,
        qqq_quotes_path=qqq_quotes_path, spy_quotes_path=spy_quotes_path,
        ledger_path=ledger_path, run_id="btc-spillover-test-d", dry_run=True))

    assert with_qqq_a["sharpe_daily_excess"] != pytest.approx(with_qqq_b["sharpe_daily_excess"])
    assert with_qqq_a["dsr_value"] != pytest.approx(with_qqq_b["dsr_value"])


def test_varying_btc_close_alone_changes_the_recorded_result(tmp_path):
    """Isolates btc_close's own causal effect: holding qqq_frame/spy_frame
    fixed, a materially different BTC price path must change the regime
    series -> the signal -> the recorded result -- proves btc_close is
    actually reaching make_signal_fn's own btc_close= kwarg, not silently
    ignored (this family's make_signal_fn returns a flat all-cash/no-op
    signal whenever btc_close is None or empty, so a mixup here would be a
    real, silent zero-trades bug, not a cosmetic one)."""
    (ledger_path, catalog_path, accepted_path,
     qqq_quotes_path, spy_quotes_path) = _write_fixture(tmp_path)
    fixed_qqq_frame = _qqq_frame()
    fixed_spy_frame = _spy_frame()

    with_btc_a = driver.compute_and_record(**_default_kwargs(
        qqq_frame=fixed_qqq_frame, spy_frame=fixed_spy_frame,
        btc_close=_btc_close(seed=33, drift=0.0002),
        catalog_path=catalog_path, accepted_path=accepted_path,
        qqq_quotes_path=qqq_quotes_path, spy_quotes_path=spy_quotes_path,
        ledger_path=ledger_path, run_id="btc-spillover-test-e", dry_run=True))

    with_btc_b = driver.compute_and_record(**_default_kwargs(
        qqq_frame=fixed_qqq_frame, spy_frame=fixed_spy_frame,
        btc_close=_btc_close(seed=90, drift=-0.0300),
        catalog_path=catalog_path, accepted_path=accepted_path,
        qqq_quotes_path=qqq_quotes_path, spy_quotes_path=spy_quotes_path,
        ledger_path=ledger_path, run_id="btc-spillover-test-f", dry_run=True))

    assert with_btc_a["n_trades"] != with_btc_b["n_trades"]


# ----------------------------------------------------------- fetch_btc_close

def _kraken_ohlc_row(time_s, close, *, open_=None, high=None, low=None, vol="1.0"):
    open_ = open_ if open_ is not None else close
    high = high if high is not None else close
    low = low if low is not None else close
    return [time_s, str(open_), str(high), str(low), str(close), str(close), vol, 3]


def _ohlc_payload(rows, result_key="XXBTZUSD"):
    body = {"last": 0}
    body[result_key] = rows
    return {"error": [], "result": body}


class _FakeResponse:
    def __init__(self, json_body, content=b"raw-bytes"):
        self._json_body = json_body
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return self._json_body


def test_fetch_btc_close_drops_an_unclosed_last_row():
    day = 86_400
    base = 1_700_000_000 - (1_700_000_000 % day)  # align to a UTC day boundary
    rows = [
        _kraken_ohlc_row(base, 100.0),
        _kraken_ohlc_row(base + day, 105.0),
        _kraken_ohlc_row(base + 2 * day, 110.0),  # still forming as of `now`
    ]
    response = _FakeResponse(_ohlc_payload(rows))
    now = datetime.fromtimestamp(base + 2 * day + 3600, tz=timezone.utc)  # 1h into the last candle

    close = driver.fetch_btc_close(
        http_get=lambda *a, **k: response, clock=lambda: now)

    assert len(close) == 2
    assert close.iloc[-1] == 105.0
    assert close.index[-1] == pd.Timestamp(base + day, unit="s")


def test_fetch_btc_close_keeps_a_closed_last_row():
    day = 86_400
    base = 1_700_000_000 - (1_700_000_000 % day)
    rows = [
        _kraken_ohlc_row(base, 100.0),
        _kraken_ohlc_row(base + day, 105.0),
    ]
    response = _FakeResponse(_ohlc_payload(rows))
    now = datetime.fromtimestamp(base + 2 * day + 3600, tz=timezone.utc)  # last candle long closed

    close = driver.fetch_btc_close(
        http_get=lambda *a, **k: response, clock=lambda: now)

    assert len(close) == 2
    assert close.iloc[-1] == 105.0
    assert close.attrs.get("last_row_closed") is True
    assert close.attrs.get("raw_sha256")
    assert close.attrs.get("retrieved_at")
