"""Regenerate the committed data file for batch5 family
``fx_carry_distribution_yield_regime``: FXA's (Invesco CurrencyShares
Australian Dollar Trust) own UNADJUSTED daily close plus its full per-share
cash-distribution history (ex-date indexed), pulled from yfinance.

    python scripts/fetch_batch5_events_fx_carry_distribution_yield_regime.py

Writes ``data/mining/batch5_events/fx_carry_distribution_yield_regime/
distributions_and_unadjusted_close.json`` and ``provenance.json``.

WHAT IT DOES
------------
Single ticker (FXA), single vendor (yfinance): ``Ticker.history(
period="max", auto_adjust=False)`` for the UNADJUSTED daily ``Close``, and
``Ticker.dividends`` for the per-share cash distribution recorded on each
ex-date. Both are written verbatim (float, 6dp), indexed by ISO date string.
The unadjusted close is what the module's yield calculation divides by (see
the module's own docstring choice 1): the pipeline's dividend-ADJUSTED close
is back-adjusted for every FUTURE distribution and would distort a trailing
yield computed over past dates.

WHAT IT DOES NOT DO -- disclosed, not silently skipped
-------------------------------------------------------
This is NOT ``preferred_stock_yield_spread_reversion``'s independent-source
reconciliation (digrin/dividendhistory/stockanalysis/nasdaq cross-checked
against Yahoo). That apparatus exists because, for THOSE three funds, Yahoo
is missing five real monthly distributions that genuinely independent
publishers list, a gap Yahoo's own two dividend endpoints share. This script
does not look for an equivalent gap in FXA's distribution history and makes
no claim about one; it is a single-vendor pull, stated as such in the
written provenance.

FXA's real distribution cadence is also NOT a strict monthly cycle the way
the preferred-stock funds are (per yfinance
``Ticker.dividends``: 2021 carries ZERO recorded distributions; 2020 carries
three, 2022 carries two -- the AUD short-term-rate carry this family times
was itself near zero over that stretch, a real economic fact, not a data
gap). A preferred_stock-style "every calendar month must carry an event"
completeness guard would therefore be WRONG for this ticker, not merely
unbuilt, so this script and the module apply only structural checks: every
recorded distribution amount must be finite and strictly positive; the close
series must be finite and strictly positive; both series must be sorted,
deduplicated, and tz-naive. A month, or a run of months, with zero
distributions is a fact this family's own rule is built to read (a
low-carry-regime signal), never an error.

Network is used inside ``main()`` only -- nothing here touches the network
at import, so ``build_payload``/``validate_payload`` are unit-testable
offline against an injected DataFrame/Series pair.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os

import pandas as pd

FAMILY_ID = "fx_carry_distribution_yield_regime"
TICKER = "FXA"

_REPO_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_DIR = os.path.join(_REPO_ROOT, "data", "mining", "batch5_events", FAMILY_ID)
DATA_PATH = os.path.join(DATA_DIR, "distributions_and_unadjusted_close.json")
PROVENANCE_PATH = os.path.join(DATA_DIR, "provenance.json")

AMOUNT_DECIMALS = 6


def _tz_naive(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_localize(None)
    return out.normalize()


def _series_to_json(series: pd.Series) -> dict:
    out = {}
    for stamp, value in series.items():
        out[pd.Timestamp(stamp).strftime("%Y-%m-%d")] = round(float(value), AMOUNT_DECIMALS)
    return out


def validate_close(close: pd.Series, name: str = "unadjusted_close") -> pd.Series:
    """Fail closed: finite, strictly positive, sorted, unique index."""
    out = pd.Series(close, copy=True)
    out.index = _tz_naive(out.index)
    out = pd.to_numeric(out, errors="coerce").astype(float)
    if out.isna().any() or not (out.to_numpy() > 0).all():
        raise ValueError(f"{name} must be finite and strictly positive throughout")
    out = out.sort_index()
    if out.index.has_duplicates:
        raise ValueError(f"{name} index must be unique")
    return out


def validate_dividends(dividends: pd.Series, name: str = "dividends") -> pd.Series:
    """Fail closed: finite, strictly positive amounts, sorted, unique index.
    A zero or negative "distribution" is never a real cash event -- and an
    absent month is a fact (see module docstring), not represented here at
    all, so there is nothing to fill or gap-check."""
    out = pd.Series(dividends, copy=True)
    out.index = _tz_naive(out.index)
    out = pd.to_numeric(out, errors="coerce").astype(float)
    if out.isna().any() or not (out.to_numpy() > 0).all():
        raise ValueError(f"{name} must be finite and strictly positive")
    out = out.sort_index()
    if out.index.has_duplicates:
        # Two distributions recorded on the same ex-date is not expected for
        # a single monthly-interest instrument; fail closed rather than
        # silently summing or dropping one.
        raise ValueError(f"{name} index must be unique")
    return out


def build_payload(close: pd.Series, dividends: pd.Series) -> dict:
    close = validate_close(close)
    dividends = validate_dividends(dividends)
    return {TICKER: {
        "unadjusted_close": _series_to_json(close),
        "dividends": _series_to_json(dividends),
    }}


def build_provenance(close: pd.Series, dividends: pd.Series, *, fetched_at: str) -> dict:
    close = validate_close(close)
    dividends = validate_dividends(dividends)
    by_year = dividends.groupby(dividends.index.year).size()
    return {
        "family_id": FAMILY_ID,
        "ticker": TICKER,
        "data_file": os.path.basename(DATA_PATH),
        "source": (
            "yfinance Ticker.history(period='max', auto_adjust=False) -- "
            "'Close' (unadjusted daily close) -- and Ticker.dividends (per-"
            "share cash distribution on its ex-date row). Single vendor: no "
            "independent cross-source reconciliation was performed (see this "
            "script's module docstring for why that is disclosed, not "
            "silently skipped)."
        ),
        "source_url": "https://finance.yahoo.com/ (via the yfinance package)",
        "date_semantics": (
            "dividend dates are EX-DIVIDEND dates (public in advance of the "
            "record date); close is the unadjusted market close, used as the "
            "yield denominator so the trailing distribution yield is the "
            "yield an investor could actually have bought at that day."
        ),
        "fetched_at": fetched_at,
        "row_count_unadjusted_close": int(len(close)),
        "event_count_dividends": int(len(dividends)),
        "distribution_events_per_year": {str(int(year)): int(count)
                                         for year, count in by_year.items()},
        "regenerated_by": "scripts/fetch_batch5_events_fx_carry_distribution_yield_regime.py",
        "note": ("regenerate with: python scripts/"
                 "fetch_batch5_events_fx_carry_distribution_yield_regime.py ; "
                 "tests use synthetic series only for arithmetic, and read "
                 "this committed file only for its real shape/coverage."),
    }


def _fetch_live():
    import yfinance as yf
    handle = yf.Ticker(TICKER)
    raw = handle.history(period="max", auto_adjust=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"{TICKER}: yfinance returned no history")
    close = raw["Close"]
    dividends = handle.dividends
    if dividends is None or len(dividends) == 0:
        raise RuntimeError(f"{TICKER}: yfinance returned no dividends")
    return close, dividends


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    close, dividends = _fetch_live()
    fetched_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = build_payload(close, dividends)
    provenance = build_provenance(close, dividends, fetched_at=fetched_at)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    with open(PROVENANCE_PATH, "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
    print(f"wrote {DATA_PATH} ({len(payload[TICKER]['unadjusted_close'])} close rows, "
          f"{len(payload[TICKER]['dividends'])} dividend events)")
    print(f"wrote {PROVENANCE_PATH}")


if __name__ == "__main__":
    main()
