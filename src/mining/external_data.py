"""Small, fail-closed clients for non-market signal data used by mining."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import os
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
# Verified via FRED's own ALFRED vintage history
# (`fred/series/observations?output_type=2`, spot-checked across
# 2015-2026): every sampled observation for these four series showed the
# IDENTICAL value across every vintage column in its window -- none has
# ever been revised after first publication. That is what makes the
# CURRENT value returned by the authenticated API also the true
# point-in-time value for these specific series; it is a property of
# THESE series, not of the endpoint or of FRED series in general (many
# FRED series -- CPIAUCSL, GDP, payrolls -- are heavily and repeatedly
# revised, sometimes years later). Never widen this set without repeating
# the same live vintage check for the new series_id.
FRED_VERIFIED_UNREVISED_SERIES = frozenset({
    "RMFSL", "WALCL", "WTREGEN", "RRPONTSYD"})
FINRA_MARGIN_STATISTICS_URL = (
    "https://www.finra.org/sites/default/files/2021-03/"
    "margin-statistics.xlsx")
FINRA_WORKSHEET = "Customer Margin Balances"
FINRA_MONTH_COLUMN = "Year-Month"
FINRA_DEBIT_COLUMN = (
    "Debit Balances in Customers' Securities Margin Accounts")
FINRA_FREE_CASH_COLUMN = "Free Credit Balances in Customers' Cash Accounts"
FINRA_FREE_SECURITIES_COLUMN = (
    "Free Credit Balances in Customers' Securities Margin Accounts")
FINRA_COLUMNS = (
    FINRA_MONTH_COLUMN,
    FINRA_DEBIT_COLUMN,
    FINRA_FREE_CASH_COLUMN,
    FINRA_FREE_SECURITIES_COLUMN,
)


@dataclass(frozen=True)
class ExternalSeriesSnapshot:
    """A parsed external series plus provenance for the exact raw vintage."""

    series: pd.Series
    metadata: dict[str, Any]


def fetch_fred_series(series_id: str, *, http_get=requests.get,
                      timeout: float = 30.0) -> pd.Series:
    """Fetch one public FRED graph CSV as a finite DatetimeIndex Series.

    FRED has used both ``DATE`` and ``observation_date`` for the first
    column, so both are accepted. Missing observations (``.``), malformed
    dates, and non-finite values are dropped. Duplicate dates keep the last
    row supplied by FRED. An empty usable result fails closed.
    """
    response = http_get(
        FRED_CSV_URL, params={"id": series_id}, timeout=timeout)
    response.raise_for_status()
    frame = pd.read_csv(io.StringIO(response.text))

    date_column = next(
        (name for name in ("DATE", "observation_date")
         if name in frame.columns), None)
    if date_column is None:
        raise ValueError(
            f"FRED {series_id}: missing DATE/observation_date column")
    if series_id not in frame.columns:
        raise ValueError(
            f"FRED {series_id}: missing required column {series_id}")
    expected_columns = {date_column, series_id}
    unexpected = [name for name in frame.columns if name not in expected_columns]
    if unexpected:
        raise ValueError(
            f"FRED {series_id}: unexpected columns {unexpected}; refusing "
            "unreviewed schema drift")

    dates = pd.to_datetime(frame[date_column], errors="coerce")
    values = pd.to_numeric(frame[series_id], errors="coerce")
    explicit_nonfinite = values.notna() & ~np.isfinite(values)
    if explicit_nonfinite.any():
        bad_rows = list(frame.index[explicit_nonfinite])
        raise ValueError(
            f"FRED {series_id}: explicit non-finite values at rows {bad_rows}")
    usable = dates.notna() & values.notna() & np.isfinite(values)
    if not usable.any():
        raise ValueError(
            f"FRED {series_id}: no usable finite observations")

    result = pd.Series(
        values.loc[usable].to_numpy(dtype=float),
        index=pd.DatetimeIndex(dates.loc[usable]),
        name=series_id,
    )
    result = result[~result.index.duplicated(keep="last")].sort_index()
    # The graph endpoint is a current-history download keyed by reference
    # dates. It provides neither historical publication timestamps nor a
    # vintage/realtime dimension. Consumers must fail closed rather than use
    # transport success as evidence that a historical ledger run is causal.
    result.attrs.update({
        "source_url": FRED_CSV_URL,
        "date_semantics": "reference_date_current_history",
        "point_in_time": False,
        "ledgerable": False,
        "ledger_block_reason": (
            "FRED graph CSV lacks point-in-time publication/vintage evidence"),
    })
    return result


def fetch_fred_series_point_in_time(
        series_id: str, *, api_key: str | None = None,
        http_get=requests.get, timeout: float = 30.0,
        clock: Callable[[], datetime] | None = None) -> pd.Series:
    """Fetch one FRED series via the authenticated observations API.

    Unlike ``fetch_fred_series`` (the public graph CSV, always marked
    non-ledgerable), this hits FRED's real ``fred/series/observations``
    endpoint -- the one that actually carries vintage/publication
    evidence -- using a free, self-service API key (``FRED_API_KEY`` env
    var, or ``api_key`` directly).

    This function still returns only the CURRENT vintage's values, never
    per-observation dynamic vintage tracking; it is marked
    ``ledgerable=True`` ONLY because ``series_id`` is on
    ``FRED_VERIFIED_UNREVISED_SERIES`` (see that constant's docstring) --
    for those specific series the current value equals the original
    first-published value, so no revision-driven look-ahead is possible.
    Callers are still responsible for applying a conservative, measured
    publication LAG (see ``signals.lag_boolean_regime_to_availability``)
    before treating any observation as tradeable -- this function supplies
    correct values only, never availability timing.
    """
    if series_id not in FRED_VERIFIED_UNREVISED_SERIES:
        raise ValueError(
            f"FRED {series_id}: not on the verified-unrevised allowlist "
            f"({sorted(FRED_VERIFIED_UNREVISED_SERIES)}) -- a series must be "
            "live-checked for revision history before this function can "
            "mark it ledgerable")
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise ValueError(
            "FRED_API_KEY not set -- pass api_key= or set the FRED_API_KEY "
            "environment variable")

    now = clock() if clock is not None else datetime.now(timezone.utc)
    response = http_get(
        FRED_API_OBSERVATIONS_URL,
        params={"series_id": series_id, "api_key": key, "file_type": "json"},
        timeout=timeout)
    response.raise_for_status()
    raw_sha256 = hashlib.sha256(bytes(response.content)).hexdigest()
    payload = response.json()
    observations = payload.get("observations") if isinstance(payload, dict) else None
    if not isinstance(observations, list):
        raise ValueError(f"FRED {series_id}: missing observations in API response")

    dates = pd.to_datetime(
        [row.get("date") for row in observations], errors="coerce")
    raw_values = pd.Series(
        [row.get("value") for row in observations], dtype=object)
    raw_values = raw_values.mask(raw_values == ".", np.nan)
    values = pd.to_numeric(raw_values, errors="coerce")
    explicit_nonfinite = values.notna() & ~np.isfinite(values)
    if explicit_nonfinite.any():
        bad_rows = list(values.index[explicit_nonfinite])
        raise ValueError(
            f"FRED {series_id}: explicit non-finite values at rows {bad_rows}")
    usable = pd.notna(dates) & values.notna() & np.isfinite(values)
    if not usable.any():
        raise ValueError(f"FRED {series_id}: no usable finite observations")

    result = pd.Series(
        values.loc[usable].to_numpy(dtype=float),
        index=pd.DatetimeIndex(dates[usable.to_numpy()]),
        name=series_id,
    )
    result = result[~result.index.duplicated(keep="last")].sort_index()
    result.attrs.update({
        "source_url": FRED_API_OBSERVATIONS_URL,
        "date_semantics": "reference_date_verified_unrevised",
        "point_in_time": True,
        "ledgerable": True,
        "ledger_block_reason": None,
        "retrieved_at": _retrieved_at_iso(now),
        "raw_sha256": raw_sha256,
    })
    return result


def _retrieved_at_iso(retrieved_at: datetime) -> str:
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise ValueError("retrieved_at must be timezone-aware")
    return (retrieved_at.astimezone(timezone.utc).isoformat()
            .replace("+00:00", "Z"))


def parse_finra_margin_debt(
        raw_workbook: bytes, *, excel_reader: Callable = pd.read_excel,
        retrieved_at: datetime, source_url: str = FINRA_MARGIN_STATISTICS_URL,
) -> ExternalSeriesSnapshot:
    """Parse the primary FINRA workbook without writing a local cache.

    The observed primary schema is intentionally exact. Any worksheet,
    column, month, or numeric drift fails closed so a layout change cannot be
    mistaken for a market signal. Values are FINRA-published USD millions and
    dates are reference month-ends, not immediate trading availability dates.
    """
    if not isinstance(raw_workbook, (bytes, bytearray)) or not raw_workbook:
        raise ValueError("FINRA workbook payload must be non-empty bytes")

    workbook = excel_reader(
        io.BytesIO(bytes(raw_workbook)), sheet_name=None, header=0)
    if not isinstance(workbook, dict) or set(workbook) != {FINRA_WORKSHEET}:
        observed = list(workbook) if isinstance(workbook, dict) else type(workbook).__name__
        raise ValueError(
            f"FINRA workbook worksheet drift: expected only "
            f"{FINRA_WORKSHEET!r}, got {observed!r}")

    frame = workbook[FINRA_WORKSHEET]
    if list(frame.columns) != list(FINRA_COLUMNS):
        raise ValueError(
            f"FINRA workbook columns drift: expected {list(FINRA_COLUMNS)!r}, "
            f"got {list(frame.columns)!r}")
    if frame.empty:
        raise ValueError("FINRA workbook has no margin-balance rows")

    month_labels = frame[FINRA_MONTH_COLUMN].astype(str).str.strip()
    if not month_labels.str.fullmatch(r"\d{4}-(0[1-9]|1[0-2])").all():
        raise ValueError("FINRA Year-Month values must use strict YYYY-MM format")
    if month_labels.duplicated().any():
        duplicates = sorted(month_labels[month_labels.duplicated(keep=False)].unique())
        raise ValueError(f"FINRA workbook has duplicate months: {duplicates}")

    raw_numeric = frame.loc[:, list(FINRA_COLUMNS[1:])]
    numeric = raw_numeric.apply(pd.to_numeric, errors="coerce")
    debit = numeric[FINRA_DEBIT_COLUMN]
    if debit.isna().any() or not np.isfinite(debit.to_numpy()).all():
        raise ValueError(
            "FINRA debit balances must contain finite numeric values")
    for column in (FINRA_FREE_CASH_COLUMN, FINRA_FREE_SECURITIES_COLUMN):
        supplied = raw_numeric[column].notna()
        supplied_values = numeric.loc[supplied, column]
        if (supplied_values.isna().any() or
                not np.isfinite(supplied_values.to_numpy()).all()):
            raise ValueError(
                "FINRA supplied free-credit balances must contain finite "
                "numeric values")

    month_ends = pd.PeriodIndex(month_labels, freq="M").to_timestamp("M")
    series = pd.Series(
        numeric[FINRA_DEBIT_COLUMN].to_numpy(dtype=float),
        index=pd.DatetimeIndex(month_ends),
        name="finra_margin_debt_usd_millions",
    ).sort_index()
    metadata = {
        "source_url": source_url,
        "retrieved_at": _retrieved_at_iso(retrieved_at),
        "raw_sha256": hashlib.sha256(bytes(raw_workbook)).hexdigest(),
        "worksheet": FINRA_WORKSHEET,
        "units": "USD millions",
        "date_semantics": "reference_month_end",
    }
    return ExternalSeriesSnapshot(series=series, metadata=metadata)


def fetch_finra_margin_debt(
        *, http_get=requests.get, excel_reader: Callable = pd.read_excel,
        timeout: float = 30.0,
        clock: Callable[[], datetime] | None = None,
) -> ExternalSeriesSnapshot:
    """Fetch and parse FINRA's primary margin-statistics workbook."""
    response = http_get(FINRA_MARGIN_STATISTICS_URL, timeout=timeout)
    response.raise_for_status()
    now = clock() if clock is not None else datetime.now(timezone.utc)
    return parse_finra_margin_debt(
        response.content, excel_reader=excel_reader, retrieved_at=now)
