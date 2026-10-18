"""Shared hermetic harness for the irreversible-path guard tests of the three
bespoke intraday ABI-1 drivers (run_vwap_cross_decaying_conviction_abi1,
run_cross_asset_divergence_fvg_abi1, run_fvg_reversal_cascade_pooled_abi1).

Everything here points at ``tmp_path``: the ledger, the catalog, the quotes
file and the research root. The network is replaced by two monkeypatches --
``intraday_data.fetch_intraday_bars`` (returns a small hand-built minute-bar
frame and records every call) and ``batch5.load_frame`` (returns a synthetic
daily benchmark frame). The real ledger (data/mining/trial_ledger.jsonl) and
the drivers' default output folders are never touched, because every path a
driver can write to is overridden explicitly and asserted on.
"""
import json
import os

import numpy as np
import pandas as pd

TZ = "America/New_York"


def _ts(date_str, time_str):
    return pd.Timestamp(f"{date_str} {time_str}", tz=TZ)


def minute_bar_frame(days=("2026-03-02", "2026-03-03", "2026-03-04")):
    """A tiny but well-formed RTH minute-bar frame (one bar at 09:30 and one
    at 09:31 per day). Enough for the real engine/bridge to run end to end;
    no entry ever fires, so no trade is produced -- these tests exercise the
    driver's guards and side effects, not the signal (covered elsewhere)."""
    rows = []
    for day in days:
        rows.append((_ts(day, "09:30"),
                     {"Open": 100.0, "High": 100.5, "Low": 99.5, "Close": 100.0,
                      "Volume": 1000.0}))
        rows.append((_ts(day, "09:31"),
                     {"Open": 100.0, "High": 100.4, "Low": 99.6, "Close": 100.1,
                      "Volume": 1000.0}))
    index = pd.DatetimeIndex([ts for ts, _ in rows], name="timestamp")
    data = {col: [r[col] for _, r in rows]
            for col in ["Open", "High", "Low", "Close", "Volume"]}
    # The real fetch also returns these two columns (intraday_data docstring).
    data["TradeCount"] = [10] * len(rows)
    data["VWAP"] = [r["Close"] for _, r in rows]
    return pd.DataFrame(data, index=index).sort_index()


def daily_benchmark_frame(start="2015-01-02", end="2026-03-04"):
    index = pd.bdate_range(start, end)
    rng = np.random.default_rng(55)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, len(index)))
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close,
         "Volume": 1_000_000}, index=index)


def _quote(bid, ask):
    return {"bid": bid, "ask": ask, "security_status": "Normal", "suspect": False}


def write_quotes(path, symbols, *, synthetic=False):
    payload = {
        "sampled_at_local": "2026-09-18T10:00:00",
        "market_state": "open",
        "source": "SYNTHETIC test capture" if synthetic else "test capture",
        "quotes": {s: _quote(600.0, 600.06) for s in symbols},
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


class Harness:
    """One hermetic run environment for a single driver module."""

    def __init__(self, tmp_path, monkeypatch, driver, *, quote_symbols,
                 catalog_tickers, catalog_parameters=None, status="eligible"):
        self.driver = driver
        self.tmp_path = tmp_path
        self.ledger_path = str(tmp_path / "trial_ledger.jsonl")
        self.catalog_path = str(tmp_path / "catalog.json")
        self.quotes_path = str(tmp_path / "quotes.json")
        self.research_root = str(tmp_path / "runs-root")
        self.fetch_calls = []
        self.load_frame_calls = []

        entry = {
            "family_id": driver.FAMILY_ID, "name": "test", "mechanism": "test",
            "tickers": list(catalog_tickers), "status": status,
            "account_types": ["taxable"], "source": "test",
            "added_date": "2026-09-18",
            "parameters": (dict(catalog_parameters) if catalog_parameters
                           is not None else dict(driver.PARAMETERS)),
        }
        with open(self.catalog_path, "w", encoding="utf-8") as handle:
            json.dump([entry], handle)
        write_quotes(self.quotes_path, quote_symbols)

        def fake_fetch(symbol, start_date, end_date, **_kwargs):
            self.fetch_calls.append((symbol, start_date, end_date))
            return minute_bar_frame()

        def fake_load_frame(symbol, **_kwargs):
            self.load_frame_calls.append(symbol)
            return daily_benchmark_frame(), "fetched"

        monkeypatch.setattr(driver.intraday_data, "fetch_intraday_bars", fake_fetch)
        monkeypatch.setattr(driver.batch5, "load_frame", fake_load_frame)
        # Never let a stale wall-clock date leak into the resolved end date.
        monkeypatch.setattr(driver, "_most_recent_available_session",
                            lambda before_date: "2026-03-04")

    def kwargs(self, **overrides):
        kwargs = dict(
            catalog_path=self.catalog_path, quotes_path=self.quotes_path,
            ledger_path=self.ledger_path, run_id="guard-test-run",
            tested_date="2026-09-18", dry_run=False, end_date=None,
            signal_source_path="scripts/test_driver.py",
            signal_source_sha256="0" * 64, research_root=self.research_root)
        kwargs.update(overrides)
        return kwargs

    def ledger_rows(self):
        if not os.path.exists(self.ledger_path):
            return []
        with open(self.ledger_path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def seed_abi1_outcome(self):
        """Pre-seed a construction=ABI-1 outcome row for the driver's family,
        through the real ledger API, so the already-scored guard can fire."""
        from mining import trial_ledger
        trial_ledger.record_outcome(
            self.ledger_path, self.driver.FAMILY_ID, "seed-run", "2026-09-01",
            stage="cut_dsr", dsr_value=0.01, sharpe_in_sample=0.1,
            ticker_label=None, construction="ABI-1", note="seeded for test")

    def cli_argv(self, *extra):
        return ["--quotes", self.quotes_path, "--ledger", self.ledger_path,
                "--catalog", self.catalog_path,
                "--research-root", self.research_root,
                "--run-id", "guard-test-run", *extra]

    def results_dir(self):
        return os.path.join(self.research_root, "guard-test-run")
