"""Regenerate the committed event/data file for batch5 family
``preferred_stock_yield_spread_reversion`` WITH an independent, cross-source
completeness reconciliation.

    python scripts/fetch_batch5_events_preferred_stock_yield_spread_reversion.py

Writes ``data/mining/batch5_events/preferred_stock_yield_spread_reversion/
distributions_and_unadjusted_close.json`` and ``provenance.json``.

WHY THIS SCRIPT EXISTS
----------------------
Comparing the Yahoo ``history()['Dividends']`` column against
``Ticker.dividends`` -- the SAME vendor's second endpoint -- is structurally
incapable of seeing a distribution the vendor never had: both endpoints agree
while Yahoo is missing FIVE real
monthly distributions (PFF 2012-11-01, PGX 2016-03-15 / 2016-04-15 /
2021-09-20, PFFD 2017-11-01), each independently listed by at least two
non-Yahoo sources. A vendor-vs-itself check is not a reconciliation, so this
script cross-checks against independent publishers and FAILS CLOSED: a month
with no distribution, for a fund that distributes monthly, is an error, never
a silent zero.

WHAT IT DOES
------------
1. Vendor pull (amount source of record, the pre-registered
   construction): ``yfinance Ticker.history(period="max", auto_adjust=False)``
   -> the UNADJUSTED ``Close`` and the ``Dividends`` column (per-share cash
   distribution on its ex-date row). ``Capital Gains`` and ``Stock Splits``
   are deliberately NOT read: the pre-registered hypothesis is a cash
   DISTRIBUTION yield, so a capital-gains distribution or a split-type
   adjustment is not a cash distribution and must never enter the trailing
   sum. (All three members report zero capital-gains rows over their full
   Yahoo history, so this exclusion is a standing guard, not a data edit.)
2. Independent pull: every source in ``INDEPENDENT_SOURCES`` that publishes
   the member (Nasdaq's dividend API serves Nasdaq-listed symbols only and
   answers "not available" for the two NYSE-Arca members -- that is recorded,
   not swallowed).
3. Reconciliation (``reconcile``), all of it fail-closed:
   * fewer than ``MIN_INDEPENDENT_SOURCES`` independent sources with data for
     a member -> error (the check cannot be performed, so it does not pass);
   * an ex-date carried by >= ``MIN_INDEPENDENT_SOURCES`` independent sources
     that the vendor lacks is MERGED IN, at the most precise amount the
     agreeing sources publish, and listed in the provenance;
   * a candidate addition within ``NEAR_DUPLICATE_DAYS`` of an event the
     vendor already has is REJECTED as a publisher date-shift (a monthly
     distributor does not pay twice inside a week; the real year-end double
     is >= three weeks apart) -- rejected, reported, never merged;
   * an addition whose ex-date is not a session in the close calendar ->
     error (it cannot be placed causally without inventing a date);
   * after merging, the calendar completeness invariants must hold (see
     ``month_gaps`` / ``incomplete_years`` / ``tail_gaps``): no unexplained
     month gap, exactly ``PAYMENTS_PER_YEAR`` events in every complete
     calendar year, AND no month missing at the END of the series -- through
     the last complete calendar month before the final close bar. That last
     one exists because the first two derive their span from the ex-dates
     themselves, so a distribution the vendor dropped at the end moves the
     span with it and hides; those newest months are the pinned holdout
     window. It mirrors the loader's own tail check, so a regenerated file
     cannot pass here and then be refused at load time.
   * amounts the vendor and >= 2 independent sources both carry are compared
     to ``AMOUNT_TOLERANCE`` (Yahoo publishes these three at 3 decimals, so
     the tolerance is one 3-decimal rounding half-step plus slack). This axis
     is REPORT-ONLY and every disagreement is written into the provenance:
     amounts are the pre-registered vendor's, so this script may surface a
     vendor rounding oddity but must not silently overwrite it, and must not
     block regeneration over it. Completeness is the axis that fails closed.

WHAT IT DOES NOT DO
-------------------
It does not re-price, re-round or re-tune anything. Vendor amounts are kept
verbatim (rounded to 6 decimals, the file's existing precision) wherever the
vendor HAS the event, so the pre-registered "yfinance Ticker.dividends" yield
construction is untouched; independent amounts are used ONLY to fill an event
the vendor is missing entirely. It writes no parameter and reads no ledger,
catalog, or holdout information.

Network is used inside ``main()`` only -- nothing here touches the network at
import, so the pure functions are unit-testable offline.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import re
import urllib.error
import urllib.request

import pandas as pd

FAMILY_ID = "preferred_stock_yield_spread_reversion"
MEMBERS = ("PFF", "PGX", "PFFD")
# Every member is a monthly distributor (verified against each fund's own
# published distribution frequency and against the independent sources).
PAYMENTS_PER_YEAR = 12
MIN_INDEPENDENT_SOURCES = 2
NEAR_DUPLICATE_DAYS = 5
# Yahoo publishes these three funds' per-share amounts at 3 decimals; an
# independent publisher at 4-6 decimals therefore differs by up to half a
# 3-decimal step. Anything beyond that is a real disagreement.
AMOUNT_TOLERANCE = 6e-4
AMOUNT_DECIMALS = 6

_REPO_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_DIR = os.path.join(_REPO_ROOT, "data", "mining", "batch5_events", FAMILY_ID)
DATA_PATH = os.path.join(DATA_DIR, "distributions_and_unadjusted_close.json")
PROVENANCE_PATH = os.path.join(DATA_DIR, "provenance.json")

_USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/120 Safari/537.36")


class ReconciliationError(RuntimeError):
    """A completeness/agreement invariant failed -- nothing is written."""


# ------------------------------------------------------- independent parsers
#
# Each parser maps one publisher's page/payload to {"YYYY-MM-DD": amount} on
# the EX-DIVIDEND date. Parsers are pure and offline-testable.

_MONEY = r"([0-9]+(?:\.[0-9]+)?)"


def parse_digrin(text: str) -> dict:
    """digrin.com stock detail page: a table of <ex-date><pay-date><amt USD>."""
    pattern = (r"<tr>\s*<td>(\d{4}-\d{2}-\d{2})</td>\s*<td>[^<]*</td>\s*<td>"
               + _MONEY + r"\s*USD")
    return {m.group(1): float(m.group(2)) for m in re.finditer(pattern, text)}


def parse_dividendhistory(text: str) -> dict:
    """dividendhistory.org payout page: <ex-date><pay-date><amount>."""
    pattern = (r"<tr[^>]*>\s*<td>(\d{4}-\d{2}-\d{2})</td>\s*<td>[\d-]*</td>\s*"
               r"<td>\D?" + _MONEY + r"</td>")
    return {m.group(1): float(m.group(2)) for m in re.finditer(pattern, text)}


def parse_stockanalysis(text: str) -> dict:
    """stockanalysis.com dividend page: an inline JS array of
    ``{dt:"...",amt:"...",...}`` records (free tier: recent years only)."""
    pattern = r'\{dt:"(\d{4}-\d{2}-\d{2})",amt:"\D?' + _MONEY + r'"'
    return {m.group(1): float(m.group(2)) for m in re.finditer(pattern, text)}


def parse_nasdaq(text: str) -> dict:
    """Nasdaq dividend API. Non-Nasdaq-listed symbols return an explicit
    "not available" message with null rows -- that is an ABSENT source
    (returns {}), which the caller must count as "no data", not as agreement.
    Only ``type == "Cash"`` rows are kept: a non-cash distribution is not a
    cash distribution yield input."""
    payload = json.loads(text)
    block = ((payload.get("data") or {}).get("dividends") or {})
    rows = block.get("rows") or []
    out = {}
    for row in rows:
        if str(row.get("type", "")).strip().lower() != "cash":
            continue
        stamp = str(row.get("exOrEffDate", "")).strip()
        if not re.fullmatch(r"\d{2}/\d{2}/\d{4}", stamp):
            continue
        month, day, year = stamp.split("/")
        amount = re.sub(r"[^0-9.]", "", str(row.get("amount", "")))
        if not amount:
            continue
        out[f"{year}-{month}-{day}"] = float(amount)
    return out


INDEPENDENT_SOURCES = {
    "digrin": {
        "url": "https://www.digrin.com/stocks/detail/{ticker}/",
        "parser": parse_digrin,
        "coverage": "full published history",
    },
    "dividendhistory": {
        "url": "https://dividendhistory.org/payout/{ticker}/",
        "parser": parse_dividendhistory,
        "coverage": "full published history",
    },
    "stockanalysis": {
        "url": "https://stockanalysis.com/etf/{lower}/dividend/",
        "parser": parse_stockanalysis,
        "coverage": "recent years only on the free tier",
    },
    "nasdaq": {
        "url": "https://api.nasdaq.com/api/quote/{ticker}/dividends?assetclass=etf",
        "parser": parse_nasdaq,
        "coverage": "Nasdaq-listed symbols only",
    },
}


# --------------------------------------------------------- calendar invariants

def _month_key(stamp) -> str:
    stamp = pd.Timestamp(stamp)
    return f"{stamp.year:04d}-{stamp.month:02d}"


def _month_range(first: str, last: str) -> list:
    year, month = int(first[:4]), int(first[5:7])
    end = (int(last[:4]), int(last[5:7]))
    out = []
    while (year, month) <= end:
        out.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return out


def _previous_month(key: str) -> str:
    year, month = int(key[:4]), int(key[5:7])
    month -= 1
    if month == 0:
        year, month = year - 1, 12
    return f"{year:04d}-{month:02d}"


def month_gaps(ex_dates) -> list:
    """Calendar months inside the series' own span that carry NO distribution
    and have no benign explanation, for a monthly distributor.

    The ONE benign pattern in these funds' real histories: a year-end
    distribution pulled forward into late December, so December carries two
    ex-dates and the following January carries none. A month whose PREVIOUS
    month has two or more ex-dates is therefore explained. Every other empty
    month is a vendor gap."""
    keys = sorted(_month_key(d) for d in ex_dates)
    if not keys:
        return []
    counts = collections.Counter(keys)
    gaps = []
    for key in _month_range(keys[0], keys[-1]):
        if counts[key]:
            continue
        if counts[_previous_month(key)] >= 2:
            continue
        gaps.append(key)
    return gaps


def _next_month(key: str) -> str:
    year, month = int(key[:4]), int(key[5:7])
    month += 1
    if month == 13:
        year, month = year + 1, 1
    return f"{year:04d}-{month:02d}"


def tail_gaps(ex_dates, last_session) -> list:
    """Calendar months missing at the END of the series: from the month after
    the last ex-date through the last COMPLETE calendar month before
    ``last_session`` (the final unadjusted-close bar).

    ``month_gaps`` derives its span from the ex-dates themselves, so a
    distribution the vendor dropped at the END moves the span end with it and
    is invisible there -- yet those newest months are exactly the pinned
    holdout window. Same pull-forward exemption as ``month_gaps``. Only
    COMPLETE months are required (the month containing ``last_session`` may
    legitimately not have paid yet), which mirrors the loader's own tail check
    in ``mining.batch5.preferred_stock_yield_spread_reversion`` -- the two
    must agree, or a regenerated file could fail at load time instead of
    here."""
    keys = sorted(_month_key(d) for d in ex_dates)
    if not keys or last_session is None:
        return []
    counts = collections.Counter(keys)
    out = []
    for key in _month_range(_next_month(keys[-1]),
                            _previous_month(_month_key(last_session))):
        if counts[key] or counts[_previous_month(key)] >= 2:
            continue
        out.append(key)
    return out


def incomplete_years(ex_dates, *, per_year: int = PAYMENTS_PER_YEAR) -> dict:
    """``{year: count}`` for every COMPLETE calendar year whose event count is
    not ``per_year``. A complete year is one strictly between the first and
    last event years, so the partial inception year and the partial current
    year are excluded (they legitimately hold fewer)."""
    years = sorted(pd.Timestamp(d).year for d in ex_dates)
    if len(years) < 2:
        return {}
    counts = collections.Counter(years)
    return {year: counts[year]
            for year in range(years[0] + 1, years[-1])
            if counts[year] != per_year}


# ------------------------------------------------------------- reconciliation

def _decimals(value: float) -> int:
    text = repr(float(value))
    return len(text.split(".")[1]) if "." in text else 0


def confirmed_missing(vendor_dates, independent: dict, *,
                      min_sources: int = MIN_INDEPENDENT_SOURCES,
                      near_duplicate_days: int = NEAR_DUPLICATE_DAYS,
                      cutoff: str | None = None) -> tuple:
    """Split the independent sources' events into
    ``(additions, rejections)``.

    ``additions`` = ``{iso_date: {"amount", "sources", "amounts"}}`` for every
    ex-date the vendor lacks that at least ``min_sources`` INDEPENDENT sources
    publish, with their amounts agreeing to ``AMOUNT_TOLERANCE``; the merged
    amount is the one the most precise agreeing source publishes.

    ``rejections`` = ``{iso_date: reason}`` for a candidate that is dropped:
    ``single_source`` (only one publisher has it -- deliberately NOT added),
    ``amount_disagreement``, ``near_duplicate_of_<date>`` (a publisher
    date-shift of an event the vendor already carries), or ``after_cutoff``
    (an announced future ex-date beyond the close series)."""
    have = {str(pd.Timestamp(d).date()) for d in vendor_dates}
    stamps = sorted(pd.Timestamp(d) for d in have)
    votes = collections.defaultdict(dict)
    for name, rows in independent.items():
        for iso, amount in rows.items():
            votes[iso][name] = float(amount)
    additions, rejections = {}, {}
    for iso in sorted(votes):
        if iso in have:
            continue
        sources = votes[iso]
        if cutoff is not None and iso > cutoff:
            rejections[iso] = "after_cutoff"
            continue
        if len(sources) < min_sources:
            rejections[iso] = "single_source"
            continue
        amounts = sorted(sources.values())
        if amounts[-1] - amounts[0] > AMOUNT_TOLERANCE:
            rejections[iso] = "amount_disagreement"
            continue
        stamp = pd.Timestamp(iso)
        near = [s for s in stamps
                if abs((s - stamp).days) <= near_duplicate_days]
        if near:
            rejections[iso] = f"near_duplicate_of_{near[0].date()}"
            continue
        best = max(sources.items(), key=lambda kv: (_decimals(kv[1]), kv[0]))
        additions[iso] = {
            "amount": round(float(best[1]), AMOUNT_DECIMALS),
            "amount_from": best[0],
            "sources": sorted(sources),
            "amounts": {name: sources[name] for name in sorted(sources)},
        }
    return additions, rejections


def amount_disagreements(vendor: pd.Series, independent: dict, *,
                         min_sources: int = MIN_INDEPENDENT_SOURCES,
                         tolerance: float = AMOUNT_TOLERANCE) -> tuple:
    """``(disagreements, max_abs_diff, compared)`` between vendor amounts and
    the CLOSEST independent amount, over ex-dates at least ``min_sources``
    independent sources also carry.

    Closest, not median, on purpose: the publishers round the same payout
    differently (one at 3 decimals, another at 4), so a median can sit between
    two rounded views of the same number and manufacture a disagreement the
    vendor is not guilty of. A vendor amount that matches ANY publisher within
    one rounding step is agreed. Report-only -- see ``reconcile``."""
    vendor_map = {str(pd.Timestamp(k).date()): float(v)
                  for k, v in vendor.items()}
    votes = collections.defaultdict(dict)
    for name, rows in independent.items():
        for iso, amount in rows.items():
            votes[iso][name] = float(amount)
    bad, worst, compared = {}, 0.0, 0
    for iso, value in sorted(vendor_map.items()):
        peers = votes.get(iso, {})
        if len(peers) < min_sources:
            continue
        compared += 1
        nearest = min(peers.items(), key=lambda kv: abs(kv[1] - value))
        diff = abs(value - nearest[1])
        worst = max(worst, diff)
        if diff > tolerance:
            bad[iso] = {"vendor": value, "nearest_source": nearest[0],
                        "nearest_independent": nearest[1],
                        "abs_diff": round(diff, 9),
                        "all_sources": {name: peers[name]
                                        for name in sorted(peers)}}
    return bad, worst, compared


def reconcile(ticker: str, vendor_dividends: pd.Series,
              close_index: pd.DatetimeIndex, independent: dict, *,
              min_sources: int = MIN_INDEPENDENT_SOURCES) -> tuple:
    """Return ``(merged_dividends, report)`` or raise ``ReconciliationError``.

    Fail-closed on: too few independent sources, an addition that is not a
    session in the close calendar, and any surviving month gap / short
    calendar year / stale tail. The amount cross-check is report-only (see
    below)."""
    if not isinstance(close_index, pd.DatetimeIndex) or len(close_index) == 0:
        raise ReconciliationError(f"{ticker}: close calendar is empty")
    with_data = {name: rows for name, rows in independent.items() if rows}
    if len(with_data) < min_sources:
        raise ReconciliationError(
            f"{ticker}: only {len(with_data)} independent source(s) returned "
            f"data ({sorted(with_data)}); {min_sources} are required -- the "
            "completeness check cannot be performed, so it does not pass")
    cutoff = str(pd.Timestamp(close_index[-1]).date())
    additions, rejections = confirmed_missing(
        vendor_dividends.index, with_data, min_sources=min_sources,
        cutoff=cutoff)
    sessions = {str(pd.Timestamp(d).date()) for d in close_index}
    off_calendar = sorted(iso for iso in additions if iso not in sessions)
    if off_calendar:
        raise ReconciliationError(
            f"{ticker}: cross-source-confirmed distribution(s) {off_calendar} "
            "are not sessions in the unadjusted close calendar -- refusing to "
            "invent a date")
    merged = pd.Series(vendor_dividends.to_numpy(dtype=float),
                       index=pd.DatetimeIndex(vendor_dividends.index))
    for iso, record in additions.items():
        merged.loc[pd.Timestamp(iso)] = record["amount"]
    merged = merged.sort_index()

    # Amount cross-check: REPORT-ONLY, deliberately. The amounts are the
    # pre-registered vendor's ("yfinance Ticker.dividends"), so this script is
    # not licensed to overwrite one; and failing the whole fetch on a
    # pre-existing vendor rounding oddity would block regeneration over
    # something it must not change. Every disagreement beyond one rounding
    # step is written into the provenance so it is visible, never silent.
    # COMPLETENESS, below, is what fails closed.
    bad_amounts, worst, compared = amount_disagreements(
        vendor_dividends, with_data, min_sources=min_sources)

    gaps = month_gaps(merged.index)
    short_years = incomplete_years(merged.index)
    tail = tail_gaps(merged.index, close_index[-1])
    if gaps or short_years or tail:
        raise ReconciliationError(
            f"{ticker}: distribution calendar is still incomplete after "
            f"cross-source merge -- unexplained empty month(s) {gaps}, "
            f"calendar year(s) not at {PAYMENTS_PER_YEAR} events "
            f"{short_years}, month(s) missing at the END of the series "
            f"(stale tail, through the last complete month before the final "
            f"price bar) {tail}. A monthly distributor with a missing month "
            "is an error, not a zero; add a source or investigate before "
            "writing.")
    report = {
        "independent_sources_used": sorted(with_data),
        "independent_source_event_counts": {name: len(rows)
                                            for name, rows in
                                            sorted(with_data.items())},
        "independent_sources_without_data": sorted(
            name for name in independent if not independent[name]),
        "vendor_event_count": int(len(vendor_dividends)),
        "merged_event_count": int(len(merged)),
        "cross_source_additions": additions,
        "candidates_rejected": rejections,
        "amount_cross_check_dates": compared,
        "amount_cross_check_max_abs_diff": round(worst, 9),
        "amount_cross_check_report_only": True,
        "amount_disagreements_beyond_rounding": bad_amounts,
        "month_gaps_after_merge": gaps,
        "calendar_years_off_expected_count_after_merge": short_years,
        "tail_gaps_after_merge": tail,
    }
    return merged, report


# --------------------------------------------------------------- network I/O

def _http_get(url: str, *, timeout: int = 45) -> str:
    request = urllib.request.Request(url, headers={
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/json,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    })
    with urllib.request.urlopen(request, timeout=timeout) as handle:
        return handle.read().decode("utf-8", "replace")


def fetch_independent(ticker: str, *, getter=_http_get) -> tuple:
    """``({source: {iso: amount}}, {source: note})`` -- a source that errors or
    does not publish the symbol comes back EMPTY with its reason recorded, so
    ``reconcile`` can refuse when too few sources answered."""
    rows, notes = {}, {}
    for name, spec in INDEPENDENT_SOURCES.items():
        url = spec["url"].format(ticker=ticker, lower=ticker.lower())
        try:
            text = getter(url)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            rows[name], notes[name] = {}, f"unavailable: {type(exc).__name__}"
            continue
        try:
            parsed = spec["parser"](text)
        except (ValueError, KeyError, TypeError) as exc:
            rows[name], notes[name] = {}, f"parse failed: {type(exc).__name__}"
            continue
        rows[name] = parsed
        notes[name] = ("no rows published for this symbol" if not parsed
                       else f"{len(parsed)} events")
    return rows, notes


def fetch_vendor(ticker: str) -> tuple:
    """``(unadjusted_close, dividends)`` from yfinance, tz-naive, normalized.
    Reads ONLY 'Close' and 'Dividends' -- see the module docstring on why
    'Capital Gains' / 'Stock Splits' are excluded."""
    import yfinance                                     # lazy: no import-time I/O
    history = yfinance.Ticker(ticker).history(period="max", auto_adjust=False)
    if history is None or history.empty:
        raise RuntimeError(f"{ticker}: yfinance returned no history")
    for column in ("Close", "Dividends"):
        if column not in history.columns:
            raise RuntimeError(f"{ticker}: yfinance history has no {column}")
    index = pd.DatetimeIndex(pd.to_datetime(history.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.normalize()
    close = pd.Series(history["Close"].to_numpy(dtype=float), index=index)
    close = close.loc[close.notna() & (close > 0)].sort_index()
    dividends = pd.Series(history["Dividends"].to_numpy(dtype=float), index=index)
    dividends = dividends.loc[dividends.notna() & (dividends > 0)].sort_index()
    return close, dividends


def vendor_self_consistency(ticker: str, dividends: pd.Series) -> dict:
    """A vendor self-consistency reading, clearly labelled as such: the
    vendor's second endpoint vs its first. It cannot detect a missing event,
    which is why completeness is established against independent publishers
    instead (see ``reconcile``)."""
    import yfinance                                     # lazy: no import-time I/O
    other = yfinance.Ticker(ticker).dividends
    index = pd.DatetimeIndex(pd.to_datetime(other.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    other = pd.Series(other.to_numpy(dtype=float), index=index.normalize())
    other = other.loc[other > 0].sort_index()
    common = dividends.index.intersection(other.index)
    return {
        "note": ("vendor-vs-same-vendor; NOT a completeness check -- see the "
                 "independent reconciliation"),
        "dividends_endpoint_events": int(len(other)),
        "dividends_endpoint_common_dates": int(len(common)),
        "dividends_endpoint_max_abs_diff": (
            float((dividends.loc[common] - other.loc[common]).abs().max())
            if len(common) else None),
    }


# ------------------------------------------------------------------- assemble

def build(members=MEMBERS, *, vendor_fetcher=fetch_vendor,
          independent_fetcher=fetch_independent,
          self_check=vendor_self_consistency) -> tuple:
    payload, provenance_members, reconciliation = {}, {}, {}
    for ticker in members:
        close, dividends = vendor_fetcher(ticker)
        independent, notes = independent_fetcher(ticker)
        merged, report = reconcile(ticker, dividends, close.index, independent)
        report["independent_source_notes"] = notes
        off = [str(d.date()) for d in merged.index if d not in close.index]
        if off:
            raise ReconciliationError(
                f"{ticker}: ex-date(s) {off} are not sessions in the close "
                "calendar")
        payload[ticker] = {
            "unadjusted_close": {str(k.date()): round(float(v), AMOUNT_DECIMALS)
                                 for k, v in close.items()},
            "dividends": {str(k.date()): round(float(v), AMOUNT_DECIMALS)
                          for k, v in merged.items()},
        }
        provenance_members[ticker] = {
            "close_first": str(close.index[0].date()),
            "close_last": str(close.index[-1].date()),
            "close_rows": int(len(close)),
            "dividend_events": int(len(merged)),
            "dividend_first": str(merged.index[0].date()),
            "dividend_last": str(merged.index[-1].date()),
            "dividend_dates_not_in_close_calendar": 0,
            "vendor_dividend_events": int(len(dividends)),
            "cross_source_additions": sorted(report["cross_source_additions"]),
            **(self_check(ticker, dividends) if self_check else {}),
        }
        reconciliation[ticker] = report
    provenance = {
        "family_id": FAMILY_ID,
        "data_file": os.path.basename(DATA_PATH),
        "fetched_at": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "regenerated_by": ("scripts/fetch_batch5_events_"
                           f"{FAMILY_ID}.py"),
        "date_semantics": (
            "dividend dates are EX-DIVIDEND dates (public weeks in advance); a "
            "value at date t is usable on/after t. Close is the unadjusted "
            "market close, used as the yield denominator so the "
            "trailing-12-month distribution yield is the yield an investor "
            "could actually have bought at that day."),
        "source": (
            "amounts and unadjusted closes: yfinance "
            "Ticker.history(period='max', auto_adjust=False) -- 'Close' "
            "(unadjusted daily close) and 'Dividends' (per-share CASH "
            "distribution on its ex-date row; 'Capital Gains' and 'Stock "
            "Splits' deliberately excluded). Completeness is established "
            "against the INDEPENDENT publishers below, never against a second "
            "endpoint of the same vendor."),
        "source_url": "https://finance.yahoo.com/ (via the yfinance package)",
        "independent_sources": {
            name: {"url_template": spec["url"], "coverage": spec["coverage"]}
            for name, spec in sorted(INDEPENDENT_SOURCES.items())},
        "reconciliation": {
            "rule": (
                "an ex-date carried by at least "
                f"{MIN_INDEPENDENT_SOURCES} independent sources that the "
                "vendor lacks is merged in at the most precise agreeing "
                "amount; a single-source event is NOT added; a candidate "
                f"within {NEAR_DUPLICATE_DAYS} days of an existing event is "
                "rejected as a publisher date-shift; after merging, every "
                "calendar month in span must carry a distribution (except a "
                "January following a two-ex-date December), every complete "
                f"calendar year must carry {PAYMENTS_PER_YEAR} events, and "
                "no month may be missing at the END of the series through "
                "the last complete calendar month before the final close "
                "bar -- otherwise the fetch FAILS and writes nothing"),
            "amount_tolerance": AMOUNT_TOLERANCE,
            "per_member": reconciliation,
        },
        "member_count": len(payload),
        "members": provenance_members,
        "row_count_unadjusted_close": sum(
            len(block["unadjusted_close"]) for block in payload.values()),
        "event_count_dividends": sum(
            len(block["dividends"]) for block in payload.values()),
        "note": ("regenerate with: python scripts/fetch_batch5_events_"
                 f"{FAMILY_ID}.py ; tests use synthetic series only"),
    }
    return payload, provenance


def _write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    os.replace(temporary, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="reconcile and print the report, write nothing")
    args = parser.parse_args(argv)
    payload, provenance = build()
    print(json.dumps(provenance["reconciliation"]["per_member"],
                     indent=1, sort_keys=True))
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    _write_json(DATA_PATH, payload)
    _write_json(PROVENANCE_PATH, provenance)
    print(f"wrote {DATA_PATH}")
    print(f"wrote {PROVENANCE_PATH}")
    return 0


if __name__ == "__main__":     # pragma: no cover
    raise SystemExit(main())
