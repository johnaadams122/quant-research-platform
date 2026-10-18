"""Cross-sectional daily panel: many tickers, one calendar, one dated scalar.

WHY THIS EXISTS. ``universe.py`` normalizes a ticker LIST and ``pooling.py``
pools TRADES across tickers (each ticker backtested separately first). Neither
can express a quantity that only exists at the market level -- how many members
are above their own 200-day average, how many advanced minus how many declined.
Breadth candidates (advance-decline line, McClellan-style oscillator,
percent-above-200dma) cannot be built without that capability.

THE DANGER THIS MODULE IS BUILT AROUND. ``backtest_engine`` already makes
TEMPORAL look-ahead structurally impossible (its signal_fn receives a slice and
reaching past it raises IndexError). Nothing in this repo addressed the
CROSS-SECTIONAL analogue: building a panel out of TODAY's index membership and
running it back decades measures a universe selected for having survived. So a
Panel's membership comes from a hand-dated membership FILE and never from the
price data, and every snapshot this module writes carries the same
``point_in_time`` / ``ledgerable`` / ``ledger_block_reason`` triple that
``external_data.fetch_fred_series`` already uses to stop non-point-in-time data
from becoming ledger evidence. A snapshot built on a basis without a vintage
dimension cannot be written here and cannot be loaded here; there is no flag.

WHY THE SEASONING RULE COUNTS BARS (a unit test reproduces both sides).
``seasoned`` must come from
membership AND a cumulative count of a ticker's OWN resolved bars, never from
``prices.rolling(window).mean().notna()``. With a rolling-derived mask,
a date on which several members have no bar produces a clean,
plausible breadth number: the missing bars NaN-propagate into the rolling mean,
those members silently leave the DENOMINATOR, and a fetch gap masquerades as a
market fact. With the membership-derived mask the same date is correctly a
broken session, its value is unknown, and it is named in the
snapshot's provenance.

FOUR DISTINCT REASONS A CELL CAN BE EMPTY, and conflating any two of them is the
failure this module exists to prevent:
  - did not exist yet          -> NOT_A_MEMBER, known from the FILE, excluded
  - existed but not seasoned   -> UNSEASONED, the statistic is undefined
  - the fetch failed           -> FETCH_FAILED, and PANEL_MAX_FETCH_FAILURES = 0
                                  makes ``build_panel`` refuse outright; a
                                  failed fetch may never become a denominator
                                  adjustment
  - seasoned member, real session, no bar -> NO_PRINT, and the WHOLE SESSION is
                                  broken. A ratio with an unknown member is
                                  unknown, not approximate.

A FIFTH WAY A DATE CAN VANISH. The four reasons above
are all per-CELL. A whole SESSION can also disappear from a reduced series for
reasons ``coverage_report``'s narrow ``broken`` column (a seasoned member with
no print) cannot see: fewer than ``min_members`` resolved, or an indicator's own
lookback leaving the first bars undefined. The caller's ``.dropna()`` removes
those rows and nothing downstream could tell -- ``AsOfSeries.as_of`` then
forward-fills the last pre-hole value across the entire omission, without bound.
For example (a unit test builds it through this module's own API), a
900-session panel with one member leaving and another joining yields a run of
about 300 CONSECUTIVE absent sessions while ``broken_session_fraction`` stays
0.0. So
``write_breadth_snapshot`` measures OMITTED sessions directly -- panel
sessions inside the snapshot's own span that carry no observation at all -- names
them in provenance, and applies PANEL_MAX_BROKEN_SESSION_FRACTION to the UNION
of broken and omitted rather than to broken alone.

PURITY. No network, no wall-clock reads. ``write_breadth_snapshot`` takes its
``built_at`` as a required argument, matching
``external_data.parse_finra_margin_debt``'s ``retrieved_at`` idiom. The only
filesystem write is the snapshot write itself: every refusal, INCLUDING the
refusal of a provenance value JSON cannot represent, happens before any byte is
written, and the write is a temp-file-plus-``os.replace`` so the destination
holds either its previous bytes or the complete new ones -- never a truncated
prefix a ledger entry might already point at.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
import hashlib
import json
import os
import tempfile

import numpy as np
import pandas as pd

from . import constants


# The file whose presence SEALS a panel cache directory against a top-up.
# fetch_universe_ohlc.fetch_and_cache is resumable by design (it skips any
# ticker whose CSV already exists), so topping up is today's DEFAULT behaviour
# -- and topping up under auto_adjust=True leaves old bars on the old
# adjustment basis and new bars on the current one, an inconsistency inside a
# single file that no existing test would catch. To extend history, fetch a new
# dated directory.
CACHE_SEAL_FILENAME = "MANIFEST.json"

# Recorded verbatim in every snapshot's provenance so a future reader can tell
# which rule produced the numbers without reading this source.
SEASONING_RULE = (
    "member AND cumulative-count-of-that-ticker's-own-resolved-bars >= window; "
    "NEVER derived from a rolling statistic")

SNAPSHOT_SCHEMA_VERSION = 1

# An advance/decline comparison needs today's close and the previous one. That
# is this indicator's own lookback, in the same sense that a 200-day
# participation statistic's lookback is 200 -- not a tunable, and deliberately
# not a pre-registered constant.
_ADVANCE_DECLINE_LOOKBACK = 2


class PanelStatus(IntEnum):
    """Why a (session, ticker) cell holds what it holds.

    The numbering is the RESOLUTION ORDER's documentation, not a ranking:
    ``resolve_status`` checks fetch failure and quarantine BEFORE membership and
    seasoning, because if a fetch failed you may not infer anything about a
    ticker's listing life from the zero bars you received.
    """

    OK = 0
    NOT_A_MEMBER = 1
    FETCH_FAILED = 2
    QUARANTINED = 3
    UNSEASONED = 4
    NO_PRINT = 5


@dataclass(frozen=True)
class Panel:
    """An aligned cross-section: sessions x tickers, plus its own provenance.

    ``prices`` holds NaN -- never 0.0 -- wherever there is no bar. A 0.00 close
    is a -100% return and an automatic "below its 200-day average", which is why
    ``pooling.py``'s ``fill_value=0.0`` union (correct for summing P&L) must
    never be copied here.

    ``member`` is derived from the membership FILE alone. That direction of
    dependency is load-bearing: derive membership from the price data and a
    fetch gap becomes indistinguishable from a delisting.

    ``status`` carries the WINDOW-INDEPENDENT statuses only (OK / NOT_A_MEMBER /
    FETCH_FAILED / QUARANTINED / NO_PRINT). UNSEASONED cannot live here because
    seasoning is a function of the consuming indicator's own lookback, and one
    Panel serves indicators with different lookbacks -- ask
    ``status_at_window(panel, window=...)`` for the full six-level frame.
    """

    prices: pd.DataFrame
    member: pd.DataFrame
    status: pd.DataFrame
    sessions: pd.DatetimeIndex
    membership: dict
    provenance: dict


class AsOfSeries:
    """Strictly-as-of scalar lookup over a dated series.

    Deliberately exposes NO index, NO slicing, NO iteration and NO
    ``__getitem__``. The only way to read a value is ``as_of(when)``, which is
    O(log n) by construction. A consumer therefore CANNOT write
    ``series.loc[:today]`` -- an O(n)-per-bar slice, which makes a whole
    backtest quadratic in its length -- and cannot write ``series.iloc[i]``
    for silent look-ahead,
    because neither surface exists. This is the structural counterpart, on the
    panel axis, of the traded axis's IndexError-on-overreach guarantee.
    """

    __slots__ = ("_dates", "_values", "_indicator", "_content_sha256")

    def __init__(self, *, dates, values, indicator: str,
                 content_sha256: str) -> None:
        index = pd.DatetimeIndex(dates)
        array = np.asarray(values, dtype=float)
        if len(index) == 0:
            raise ValueError(
                "AsOfSeries: refusing an empty series -- an indicator with no "
                "observations is a build failure, not a valid snapshot")
        if len(index) != len(array):
            raise ValueError(
                f"AsOfSeries: dates ({len(index)}) and values ({len(array)}) "
                "must be the same length")
        if index.tz is not None:
            raise ValueError("AsOfSeries: dates must be timezone-naive")
        if not index.is_monotonic_increasing or index.has_duplicates:
            raise ValueError("AsOfSeries: dates must be sorted and unique")
        if not np.isfinite(array).all():
            raise ValueError(
                "AsOfSeries: values must all be finite -- a broken or thin "
                "session is DROPPED from a snapshot, never carried as NaN")
        if not isinstance(indicator, str) or not indicator:
            raise ValueError("AsOfSeries: indicator must be a non-empty string")
        if not isinstance(content_sha256, str) or not content_sha256:
            raise ValueError(
                "AsOfSeries: content_sha256 must be a non-empty string")
        object.__setattr__(self, "_dates", index.values)
        object.__setattr__(self, "_values", array)
        object.__setattr__(self, "_indicator", indicator)
        object.__setattr__(self, "_content_sha256", content_sha256)

    def as_of(self, when) -> float | None:
        """Last value dated at or before ``when``; None before the first one.

        A date the snapshot DROPPED (broken or thin) is indistinguishable from a
        non-session at lookup time by design -- the honest record of which dates
        were dropped lives in the snapshot's provenance, in BOTH
        ``broken_sessions`` and ``omitted_sessions``. That second list exists
        because this method forward-fills across a hole without bound, so a hole
        that provenance does not name is a hole no consumer can detect; see the
        module docstring's fifth-way note. The consuming signal factory's
        contract is therefore "no NEW decision on an unknown day", never a
        fabricated value.
        """
        stamp = pd.Timestamp(when)
        if stamp.tz is not None:
            raise ValueError(
                "AsOfSeries.as_of: query timestamp must be timezone-naive; the "
                "mining pipeline normalizes market calendars before backtesting")
        position = int(np.searchsorted(
            self._dates, stamp.to_datetime64(), side="right")) - 1
        if position < 0:
            return None
        return float(self._values[position])

    @property
    def indicator(self) -> str:
        return self._indicator

    @property
    def content_sha256(self) -> str:
        return self._content_sha256

    def __len__(self) -> int:
        return int(len(self._dates))

    def describe(self) -> dict:
        """Reporting only -- never a data path."""
        return {
            "indicator": self._indicator,
            "content_sha256": self._content_sha256,
            "n": len(self),
            "first_date": str(pd.Timestamp(self._dates[0]).date()),
            "last_date": str(pd.Timestamp(self._dates[-1]).date()),
        }


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------

def sessions_between(start: str, end: str, *,
                     calendar_factory=None) -> pd.DatetimeIndex:
    """NYSE sessions in [start, end] as a tz-naive, sorted, unique index.

    ``calendar_factory`` is injected so a unit test never has to depend on the
    packaged exchange calendar; the default builds the NYSE calendar from
    pandas_market_calendars (already a requirement of this repo, and verified
    to reach back to 1962-01-02).
    """
    first = pd.Timestamp(start)
    last = pd.Timestamp(end)
    if first > last:
        raise ValueError(
            f"sessions_between: start {first.date()} is after end {last.date()}")
    if calendar_factory is None:
        import pandas_market_calendars as mcal  # local: keeps import cheap

        def calendar_factory():
            return mcal.get_calendar("NYSE")

    days = pd.DatetimeIndex(calendar_factory().valid_days(
        start_date=str(first.date()), end_date=str(last.date())))
    if days.tz is not None:
        days = days.tz_convert("UTC").tz_localize(None)
    days = pd.DatetimeIndex(days).normalize().unique().sort_values()
    if len(days) == 0:
        raise ValueError(
            f"sessions_between: no market sessions between {first.date()} and "
            f"{last.date()} -- refusing to build a panel over an empty calendar")
    return days


# ---------------------------------------------------------------------------
# Reference files
# ---------------------------------------------------------------------------

def _read_json(path: str, label: str) -> dict:
    if not os.path.exists(path):
        raise ValueError(f"{label}: file not found at {path!r}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label}: {path!r} must contain a JSON object")
    return payload


def _validate_membership(membership: dict) -> None:
    if not isinstance(membership, dict):
        raise ValueError("membership must be a dict")
    for key in ("basis_id", "point_in_time", "ledgerable", "members"):
        if key not in membership:
            raise ValueError(f"membership: missing required key {key!r}")
    if not isinstance(membership["point_in_time"], bool):
        raise ValueError("membership: point_in_time must be a bool")
    if not isinstance(membership["ledgerable"], bool):
        raise ValueError("membership: ledgerable must be a bool")
    members = membership["members"]
    if not isinstance(members, list) or not members:
        raise ValueError(
            "membership: 'members' is empty -- an empty membership basis is a "
            "load/parse failure, not a real 0-member index "
            "(same rule as universe.build_universe)")
    seen = set()
    for position, entry in enumerate(members):
        if not isinstance(entry, dict) or not entry.get("ticker"):
            raise ValueError(
                f"membership: member {position} has no ticker")
        ticker = entry["ticker"]
        if ticker in seen:
            raise ValueError(
                f"membership: duplicate member ticker {ticker!r} -- a member "
                "counted twice silently doubles its weight in every ratio")
        seen.add(ticker)
        start = entry.get("start")
        if start is None:
            raise ValueError(
                f"membership: member {ticker!r} has no start date; membership "
                "must be knowable on every date, never inferred from prices")
        try:
            start_stamp = pd.Timestamp(start)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"membership: member {ticker!r} has an unparseable start "
                f"{start!r}") from exc
        if pd.isna(start_stamp):
            raise ValueError(
                f"membership: member {ticker!r} has an unparseable start "
                f"{start!r}")
        end = entry.get("end")
        if end is not None:
            try:
                end_stamp = pd.Timestamp(end)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"membership: member {ticker!r} has an unparseable end "
                    f"{end!r}") from exc
            if pd.isna(end_stamp) or end_stamp < start_stamp:
                raise ValueError(
                    f"membership: member {ticker!r} ends {end!r} before it "
                    f"starts {start!r}")


def load_membership(path: str) -> dict:
    """Load and fully validate a hand-dated point-in-time membership file."""
    membership = _read_json(path, "load_membership")
    _validate_membership(membership)
    return membership


def load_panel_manifest(path: str) -> dict:
    """Load and validate a panel fetch manifest (the cache's attestation)."""
    manifest = _read_json(path, "load_panel_manifest")
    for key in ("schema_version", "basis_id", "adjustment", "per_ticker",
                "failed"):
        if key not in manifest:
            raise ValueError(
                f"load_panel_manifest: {path!r} is missing required key {key!r}")
    if not isinstance(manifest["per_ticker"], dict):
        raise ValueError("load_panel_manifest: 'per_ticker' must be an object")
    if not isinstance(manifest["failed"], dict):
        raise ValueError("load_panel_manifest: 'failed' must be an object")
    if manifest["adjustment"] != constants.PANEL_PRICE_ADJUSTMENT:
        raise ValueError(
            f"load_panel_manifest: adjustment {manifest['adjustment']!r} is not "
            f"the pre-registered {constants.PANEL_PRICE_ADJUSTMENT!r}; two "
            "adjustment bases must never share a panel")
    return manifest


def assert_cache_dir_unsealed(cache_dir: str) -> None:
    """Refuse to treat a sealed panel cache directory as writable.

    Read-only: this only inspects the directory. The fetch script calls it
    before writing anything, so extending history means a NEW dated directory
    rather than a top-up that mixes two adjustment bases inside one file.
    """
    seal = os.path.join(cache_dir, CACHE_SEAL_FILENAME)
    if os.path.exists(seal):
        raise ValueError(
            f"panel cache directory {cache_dir!r} is SEALED: it already "
            f"contains {CACHE_SEAL_FILENAME}. Topping up a sealed directory "
            "would leave old bars on the old adjustment basis and new bars on "
            "the current one, inside the same file, with no error raised "
            "anywhere. Fetch a NEW dated directory instead.")


def load_cached_closes(cache_dir: str, tickers: list, *, manifest_path: str,
                       field: str = "Close") -> dict:
    """Read per-ticker cached closes, verifying every CSV against the manifest."""
    manifest = load_panel_manifest(manifest_path)
    failed = manifest["failed"]
    if failed:
        raise ValueError(
            f"load_cached_closes: manifest {manifest_path!r} reports failed "
            f"fetches {sorted(failed)}; PANEL_MAX_FETCH_FAILURES is "
            f"{constants.PANEL_MAX_FETCH_FAILURES}, and a failed member "
            "corrupts every date's denominator rather than costing one ticker's "
            "trades. Re-fetch into a new dated directory.")

    per_ticker = manifest["per_ticker"]
    out = {}
    for ticker in tickers:
        if ticker not in per_ticker:
            raise ValueError(
                f"load_cached_closes: ticker {ticker!r} is absent from manifest "
                f"{manifest_path!r} -- an unattested CSV is not evidence")
        csv_path = os.path.join(cache_dir, f"{ticker}.csv")
        if not os.path.exists(csv_path):
            raise ValueError(
                f"load_cached_closes: ticker {ticker!r} has no cached CSV at "
                f"{csv_path!r}")
        with open(csv_path, "rb") as handle:
            raw = handle.read()
        observed = hashlib.sha256(raw).hexdigest()
        expected = per_ticker[ticker].get("sha256")
        if observed != expected:
            raise ValueError(
                f"load_cached_closes: ticker {ticker!r} CSV sha256 {observed} "
                f"does not match the manifest's {expected}; the cache and its "
                "attestation disagree")
        frame = pd.read_csv(csv_path, index_col=0, parse_dates=[0])
        if field not in frame.columns:
            raise ValueError(
                f"load_cached_closes: ticker {ticker!r} CSV has no {field!r} "
                f"column (found {list(frame.columns)})")
        series = frame[field].astype(float)
        series.index = pd.DatetimeIndex(series.index)
        series.name = ticker
        _validate_price_series(ticker, series)
        rows = per_ticker[ticker].get("rows")
        if rows is not None and int(rows) != len(series):
            raise ValueError(
                f"load_cached_closes: ticker {ticker!r} has {len(series)} rows "
                f"but the manifest attests {rows}")
        out[ticker] = series
    return out


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

def _validate_price_series(ticker: str, series: pd.Series) -> None:
    """Per-series validation lifted in spirit from rotation_engine._validate_frame.

    Its ``index.equals(expected_calendar)`` rule is deliberately NOT carried
    over: a panel's members have differing listing histories by construction
    (on the sector_spdr_11 basis, three distinct inception dates), so identical
    calendars is an unsatisfiable requirement rather than a safety property.
    """
    if not isinstance(series, pd.Series) or series.empty:
        raise ValueError(f"prices for {ticker!r} must be a non-empty Series")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(f"prices for {ticker!r} must use a DatetimeIndex")
    if series.index.tz is not None:
        raise ValueError(
            f"prices for {ticker!r} must be timezone-naive; the mining "
            "pipeline normalizes market calendars before backtesting")
    if not series.index.is_monotonic_increasing or series.index.has_duplicates:
        raise ValueError(f"prices for {ticker!r} must be sorted unique by date")
    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"prices for {ticker!r} must all be finite")
    if (values <= 0.0).any():
        raise ValueError(
            f"prices for {ticker!r} must be positive -- a 0.00 close is a -100% "
            "return and an automatic 'below its moving average'")


def membership_mask(membership: dict, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """True iff the membership FILE covers that (session, ticker) cell.

    Derived from the file alone. Never from the price data -- that direction of
    dependency is what makes a fetch gap distinguishable from a delisting.
    """
    _validate_membership(membership)
    index = pd.DatetimeIndex(sessions)
    tickers = sorted(entry["ticker"] for entry in membership["members"])
    mask = pd.DataFrame(False, index=index, columns=tickers)
    for entry in membership["members"]:
        start = pd.Timestamp(entry["start"])
        end = None if entry.get("end") is None else pd.Timestamp(entry["end"])
        covered = index >= start
        if end is not None:
            covered = covered & (index <= end)
        mask.loc[:, entry["ticker"]] = covered
    return mask


def resolve_status(*, prices: pd.DataFrame, member: pd.DataFrame,
                   fetch_failed=frozenset(), quarantined=frozenset(),
                   seasoned: pd.DataFrame | None = None) -> pd.DataFrame:
    """Resolve every cell's PanelStatus. THE ORDER IS THE GUARANTEE.

      1. ticker failed to fetch  -> FETCH_FAILED for EVERY session
      2. ticker is quarantined   -> QUARANTINED for every session
      3. not a member            -> NOT_A_MEMBER
      4. member but not seasoned -> UNSEASONED   (only when ``seasoned`` given)
      5. seasoned member, no bar -> NO_PRINT
      6. otherwise               -> OK

    Steps 1-2 precede 3-5 deliberately: if a fetch failed you may not infer
    anything about a ticker's listing life from the zero bars you received.
    Reversing that order relabels a failed ticker's early sessions UNSEASONED and
    its pre-membership sessions NOT_A_MEMBER, i.e. it manufactures a listing
    history out of a transport failure.

    ``seasoned=None`` yields the window-independent base status; UNSEASONED then
    never appears, because seasoning belongs to the consuming indicator's own
    lookback rather than to the Panel.
    """
    columns = list(member.columns)
    aligned = prices.reindex(index=member.index, columns=columns)
    status = pd.DataFrame(
        int(PanelStatus.OK), index=member.index, columns=columns, dtype="int8")

    has_print = aligned.notna()
    is_member = member.astype(bool)
    if seasoned is None:
        is_seasoned = is_member
    else:
        is_seasoned = seasoned.reindex(
            index=member.index, columns=columns).astype(bool)

    # Applied in REVERSE precedence so the highest-precedence rule is written
    # last and therefore wins every cell it claims.
    status = status.mask(is_member & is_seasoned & ~has_print,
                         int(PanelStatus.NO_PRINT))
    status = status.mask(is_member & ~is_seasoned, int(PanelStatus.UNSEASONED))
    status = status.mask(~is_member, int(PanelStatus.NOT_A_MEMBER))
    for ticker in quarantined:
        if ticker in status.columns:
            status.loc[:, ticker] = int(PanelStatus.QUARANTINED)
    for ticker in fetch_failed:
        if ticker in status.columns:
            status.loc[:, ticker] = int(PanelStatus.FETCH_FAILED)
    return status.astype("int8")


def build_panel(*, prices_by_ticker: dict, membership: dict,
                sessions: pd.DatetimeIndex,
                fetch_failed=frozenset(), quarantined=frozenset(),
                provenance: dict) -> Panel:
    """Align per-ticker closes onto one session calendar. Fails closed."""
    _validate_membership(membership)
    if not isinstance(provenance, dict):
        raise ValueError("build_panel: provenance must be a dict")

    index = pd.DatetimeIndex(sessions)
    if len(index) == 0:
        raise ValueError("build_panel: sessions is empty")
    if index.tz is not None:
        raise ValueError("build_panel: sessions must be timezone-naive")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("build_panel: sessions must be sorted and unique")

    tickers = sorted(entry["ticker"] for entry in membership["members"])
    fetch_failed = frozenset(fetch_failed)
    quarantined = frozenset(quarantined)

    unknown = sorted((fetch_failed | quarantined) - set(tickers))
    if unknown:
        raise ValueError(
            f"build_panel: fetch_failed/quarantined names {unknown} are not "
            "members of this basis")

    # The session calendar must cover every member's membership span; silently
    # truncating a member's history would change the denominator without saying
    # so.
    for entry in membership["members"]:
        start = pd.Timestamp(entry["start"])
        if start < index[0] or start > index[-1]:
            raise ValueError(
                f"build_panel: member {entry['ticker']!r} starts {start.date()}, "
                f"outside the session calendar {index[0].date()}..."
                f"{index[-1].date()}")

    needed = set(tickers) - fetch_failed
    missing = sorted(needed - set(prices_by_ticker))
    if missing:
        raise ValueError(
            f"build_panel: no price series supplied for members {missing}")
    extra = sorted(set(prices_by_ticker) - set(tickers))
    if extra:
        raise ValueError(
            f"build_panel: price series supplied for non-members {extra}")

    span_lo, span_hi = index[0], index[-1]
    columns = {}
    for ticker in tickers:
        if ticker in fetch_failed or ticker in quarantined:
            # No bar may be trusted from a failed or quarantined ticker, and a
            # quarantined ticker's data is discarded even when present.
            columns[ticker] = pd.Series(np.nan, index=index, dtype=float)
            continue
        series = prices_by_ticker[ticker]
        _validate_price_series(ticker, series)
        in_span = series.index[(series.index >= span_lo) & (series.index <= span_hi)]
        off_calendar = in_span.difference(index)
        if len(off_calendar) > 0:
            raise ValueError(
                f"build_panel: ticker {ticker!r} has bars on "
                f"{[str(d.date()) for d in off_calendar[:5]]} which are not "
                "sessions on this calendar -- the price data and the session "
                "calendar disagree")
        columns[ticker] = series.reindex(index).astype(float)

    prices = pd.DataFrame(columns, index=index, columns=tickers)
    member = membership_mask(membership, index)
    status = resolve_status(
        prices=prices, member=member, fetch_failed=fetch_failed,
        quarantined=quarantined, seasoned=None)

    if len(fetch_failed) > constants.PANEL_MAX_FETCH_FAILURES:
        raise ValueError(
            f"build_panel: {len(fetch_failed)} member(s) failed to fetch "
            f"{sorted(fetch_failed)}, above PANEL_MAX_FETCH_FAILURES="
            f"{constants.PANEL_MAX_FETCH_FAILURES}. In a cross-sectional "
            "statistic a failed member corrupts every date's denominator for "
            "the whole panel, so it may never become a denominator adjustment.")

    return Panel(prices=prices, member=member, status=status, sessions=index,
                 membership=membership, provenance=dict(provenance))


# ---------------------------------------------------------------------------
# Seasoning, coverage, broken sessions
# ---------------------------------------------------------------------------

def _check_window(window: int, name: str = "window") -> int:
    if not isinstance(window, (int, np.integer)) or isinstance(window, bool):
        raise ValueError(f"{name} must be an int, got {window!r}")
    window = int(window)
    if window < 1:
        raise ValueError(f"{name} must be >= 1, got {window}")
    return window


def seasoned_mask(panel: Panel, *, window: int) -> pd.DataFrame:
    """member AND that ticker's own cumulative resolved-bar count >= window.

    NEVER ``prices.rolling(window).mean().notna()``. See this module's docstring
    for why: with the rolling form, a date on which several members have no
    bar produces a clean, plausible breadth number because the missing
    members silently leave the denominator.
    """
    window = _check_window(window)
    resolved = panel.member.astype(bool) & panel.prices.notna()
    return panel.member.astype(bool) & (resolved.cumsum() >= window)


def broken_sessions(panel: Panel, *, window: int) -> pd.Series:
    """True on any session where a SEASONED member has no bar.

    That session's breadth value is genuinely unknown -- not approximate -- so it
    is dropped rather than estimated, and named individually in the snapshot's
    provenance.
    """
    seasoned = seasoned_mask(panel, window=window)
    return (seasoned & panel.prices.isna()).any(axis=1)


def require_no_broken_sessions(panel: Panel, *, window: int,
                               start=None, end=None) -> None:
    """Raise unless ZERO sessions are broken across [start, end].

    A path-dependent consumer cannot use "a low broken fraction": pandas
    ``.cumsum()`` and ``.ewm()`` SKIP NaN, so a gap is bridged invisibly and the
    contamination carries to every subsequent date rather than to one bar. Such
    a consumer needs a zero-gap guarantee over the exact span it accumulates
    over, which is what this provides.
    """
    broken = broken_sessions(panel, window=window)
    if start is not None:
        broken = broken.loc[broken.index >= pd.Timestamp(start)]
    if end is not None:
        broken = broken.loc[broken.index <= pd.Timestamp(end)]
    offenders = [str(stamp.date()) for stamp in broken.index[broken.to_numpy()]]
    if offenders:
        shown = offenders if len(offenders) <= 25 else (
            offenders[:25] + [f"...and {len(offenders) - 25} more"])
        raise ValueError(
            f"require_no_broken_sessions: {len(offenders)} broken session(s) in "
            f"span: {shown}. Every one has at least one seasoned member with no "
            "bar, so its breadth value is unknown; .cumsum()/.ewm() would skip "
            "it silently and bridge the gap.")


def coverage_report(panel: Panel, *, window: int) -> pd.DataFrame:
    """Per-session member accounting: expected / resolved / seasoned / broken."""
    seasoned = seasoned_mask(panel, window=window)
    has_print = panel.prices.notna()
    return pd.DataFrame({
        "expected_n": panel.member.astype(bool).sum(axis=1).astype("int64"),
        "resolved_n": (seasoned & has_print).sum(axis=1).astype("int64"),
        "seasoned_n": seasoned.sum(axis=1).astype("int64"),
        "broken": (seasoned & ~has_print).any(axis=1),
    }, index=panel.sessions)


def status_at_window(panel: Panel, *, window: int) -> pd.DataFrame:
    """The full six-level status frame for one indicator's lookback."""
    return resolve_status(
        prices=panel.prices, member=panel.member,
        fetch_failed=frozenset(
            panel.status.columns[
                (panel.status == int(PanelStatus.FETCH_FAILED)).all(axis=0)]),
        quarantined=frozenset(
            panel.status.columns[
                (panel.status == int(PanelStatus.QUARANTINED)).all(axis=0)]),
        seasoned=seasoned_mask(panel, window=window))


# ---------------------------------------------------------------------------
# Breadth reductions: Panel -> one dated scalar series
# ---------------------------------------------------------------------------

def _reduce(known: pd.DataFrame, seasoned: pd.DataFrame,
            numerator: pd.Series, min_members: int, name: str) -> pd.Series:
    """Shared tail of every pointwise reduction.

    Two independent NaN rules, and both matter:
      - ANY seasoned member whose value is unknown makes the WHOLE session
        unknown. Dropping that member from the denominator instead is precisely
        the masquerade this module exists to prevent.
      - fewer than ``min_members`` resolved members makes the session unknown
        too, on the >=-passes convention this repo already uses for
        DSR_PASS_THRESHOLD and MIN_TRADES_FOR_SCORING.
    """
    if not isinstance(min_members, (int, np.integer)) or isinstance(min_members, bool):
        raise ValueError(f"min_members must be an int, got {min_members!r}")
    min_members = int(min_members)
    if min_members < 1:
        raise ValueError(f"min_members must be >= 1, got {min_members}")

    resolved_n = known.sum(axis=1)
    value = numerator / resolved_n.where(resolved_n > 0)
    value = value.mask((seasoned & ~known).any(axis=1))
    value = value.mask(resolved_n < min_members)
    value.name = name
    return value.astype(float)


def _trailing_mean_over_own_bars(prices: pd.DataFrame,
                                 window: int) -> pd.DataFrame:
    """Each ticker's trailing ``window``-BAR mean, over that ticker's OWN bars.

    Deliberately NOT ``prices.rolling(window).mean()`` over the session frame.
    A frame-level rolling mean is NaN for ``window - 1`` sessions after ANY gap
    in that ticker's column, so one missing print would blank a healthy member's
    reading for months afterwards. Averaging over the ticker's own resolved bars
    is also the only definition COHERENT with the seasoning rule the mask is
    built from ("cumulative count of that ticker's own resolved bars >=
    window"): a cell the mask calls seasoned then always has a defined mean, so
    the two rules cannot disagree. The sibling reducer
    ``mining/panel_signals.py`` defines it identically -- the same statistic must
    not have two definitions in one repo.

    Cost note: this is one dropna+rolling pass per COLUMN per call, not per bar.
    It is not a full-frame recompute inside a per-bar callback, which would be
    quadratic over a backtest.
    """
    columns = {}
    for ticker in prices.columns:
        column = prices[ticker]
        own = column.dropna()
        columns[ticker] = own.rolling(
            window, min_periods=window).mean().reindex(column.index)
    return pd.DataFrame(columns, index=prices.index, columns=prices.columns)


def breadth_participation(panel: Panel, *, window: int,
                          min_members: int) -> pd.Series:
    """Fraction of seasoned, resolved members trading above their own window-mean.

    ``>`` is strict: a price exactly on its own mean is not a participant. A
    seasoned member whose mean is somehow undefined is UNKNOWN, and one unknown
    seasoned member makes the WHOLE session unknown -- it is never counted as a
    decliner and never quietly dropped from the denominator.
    """
    window = _check_window(window)
    seasoned = seasoned_mask(panel, window=window)
    prices = panel.prices
    mean = _trailing_mean_over_own_bars(prices, window)
    known = seasoned & prices.notna() & mean.notna()
    above = (prices > mean) & known
    return _reduce(known, seasoned, above.sum(axis=1), min_members,
                   "participation_above_ma")


def breadth_net_advance_ratio(panel: Panel, *, min_members: int) -> pd.Series:
    """(advancers - decliners) / resolved members, per session.

    The RATIO form is mandatory rather than stylistic: a panel's denominator
    genuinely changes as members join (9 -> 10 -> 11 on the sector_spdr_11
    basis), and a raw net-advance COUNT would carry a level shift across every
    membership change that no downstream threshold could distinguish from a
    market fact.
    """
    seasoned = seasoned_mask(panel, window=_ADVANCE_DECLINE_LOOKBACK)
    prices = panel.prices
    previous = prices.shift(1)
    known = seasoned & prices.notna() & previous.notna()
    advancers = ((prices > previous) & known).sum(axis=1)
    decliners = ((prices < previous) & known).sum(axis=1)
    return _reduce(known, seasoned, advancers - decliners, min_members,
                   "net_advance_ratio")


def _accumulation_span(series: pd.Series, label: str) -> pd.Series:
    """Trim leading pre-history NaNs, then REFUSE any interior gap.

    Leading NaNs are honest pre-history (members not yet seasoned). An interior
    NaN is a genuinely unknown session, and pandas ``.cumsum()``/``.ewm()`` skip
    NaN -- so accumulating across one bridges the gap silently and shifts every
    subsequent value, rather than failing.
    """
    valid = series.notna().to_numpy()
    if not valid.any():
        raise ValueError(
            f"{label}: the net-advance series has no valid observations at all")
    first = int(np.argmax(valid))
    span = series.iloc[first:]
    gaps = [str(stamp.date()) for stamp in span.index[span.isna().to_numpy()]]
    if gaps:
        shown = gaps if len(gaps) <= 25 else gaps[:25] + [
            f"...and {len(gaps) - 25} more"]
        raise ValueError(
            f"{label}: refusing to accumulate across {len(gaps)} unknown "
            f"session(s) {shown}. pandas .cumsum()/.ewm() SKIP NaN, so the gap "
            "would be bridged invisibly and every subsequent value shifted. "
            "Re-fetch the missing bars into a new dated cache directory.")
    return span


def breadth_net_advance_oscillator(panel: Panel, *, fast: int, slow: int,
                                   min_members: int) -> pd.Series:
    """EMA_fast(net advance ratio) - EMA_slow(...). Path-dependent: no gaps."""
    fast = _check_window(fast, "fast")
    slow = _check_window(slow, "slow")
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be shorter than slow ({slow})")
    ratio = breadth_net_advance_ratio(panel, min_members=min_members)
    span = _accumulation_span(ratio, "breadth_net_advance_oscillator")
    value = (span.ewm(span=fast, adjust=False).mean()
             - span.ewm(span=slow, adjust=False).mean())
    value = value.reindex(panel.sessions)
    value.name = "net_advance_ratio_oscillator"
    return value


def breadth_cumulative_ad_trend(panel: Panel, *, trend_days: int,
                                min_members: int) -> pd.Series:
    """cumsum(net advance ratio) minus its own trailing mean. No gaps allowed."""
    trend_days = _check_window(trend_days, "trend_days")
    ratio = breadth_net_advance_ratio(panel, min_members=min_members)
    span = _accumulation_span(ratio, "breadth_cumulative_ad_trend")
    cumulative = span.cumsum()
    value = cumulative - cumulative.rolling(
        trend_days, min_periods=trend_days).mean()
    value = value.reindex(panel.sessions)
    value.name = "cumulative_ad_trend"
    return value


# ---------------------------------------------------------------------------
# Snapshots: the bytes a recorded ledger verdict rests on
# ---------------------------------------------------------------------------

def _canonical_sha256(*, schema_version: int, indicator: str, parameters: dict,
                      dates: list, values: list, provenance: dict) -> str:
    """Hash EVERYTHING the file asserts, not merely the numbers.

    ``provenance`` and ``schema_version`` are inside the payload deliberately.
    If the hash covered only indicator/parameters/dates/values, a snapshot
    whose ``broken_sessions`` were rewritten to ``['FORGED']``, its
    ``seasoning_rule`` to arbitrary text and its ``built_at`` to 1999 would
    still load against its originally-pinned sha: every claim about HOW the
    numbers were produced would be unprotected -- including the
    broken/omitted session lists that are the only durable record of data
    holes. ``sort_keys=True`` makes the
    provenance dict's ordering irrelevant, so the hash is stable across rebuilds
    of the same content.
    """
    payload = json.dumps(
        {"schema_version": schema_version, "indicator": indicator,
         "parameters": parameters, "dates": dates, "values": values,
         "provenance": provenance},
        sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_json_serializable(mapping: dict, label: str) -> None:
    """Refuse a mapping JSON cannot represent, NAMING the offending key.

    ``json.dump`` raises only once it is already writing, i.e. after ``open``
    has truncated the destination -- so a provenance dict holding a
    ``pd.Timestamp`` (an entirely ordinary thing for a provenance dict about
    a fetch to carry) would surface as a ``TypeError`` plus a truncated JSON
    file at the target path. This check prevents both: the refusal is a
    ValueError naming the offending field, per this repo's fail-closed rule,
    and it happens before any byte is written.

    ``sort_keys=True`` matches ``_canonical_sha256``'s own serialization
    exactly, so a mapping that passes here cannot fail there.
    """
    if not isinstance(mapping, dict):
        raise ValueError(f"{label} must be a dict")
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise ValueError(
                f"{label}: key {key!r} is a {type(key).__name__}, not a string. "
                "json.dump would silently coerce it, so the bytes on disk would "
                "not be the object that was hashed.")
        try:
            json.dumps({key: value}, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label}: key {key!r} holds a {type(value).__name__} that "
                f"cannot be written as JSON ({exc}). Convert it at the call "
                "site -- a snapshot that cannot be serialized is refused "
                "before any file is touched, never half-written.") from exc


def _iso_utc(stamp: datetime) -> str:
    if not isinstance(stamp, datetime):
        raise ValueError("built_at must be a datetime")
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("built_at must be timezone-aware")
    return (stamp.astimezone(timezone.utc).isoformat()
            .replace("+00:00", "Z"))


def write_breadth_snapshot(series: pd.Series, path: str, *, indicator: str,
                           panel: Panel, parameters: dict,
                           seasoning_window: int, path_dependent: bool,
                           built_at: datetime) -> dict:
    """Persist a reduced breadth series with the provenance it stands on.

    Every refusal below happens BEFORE any byte is written, and the write itself
    is a temp file plus ``os.replace``, so a rejected or crashed build leaves the
    destination byte-identical to whatever it already was:
      - a membership basis that is not point-in-time / not ledgerable cannot be
        written at all, under any argument. There is no research hatch: an
        unmetered search over data that will later BE the holdout is exactly
        what this pipeline exists to prevent.
      - a NaN anywhere in ``series`` (a broken or thin session must be DROPPED,
        never carried as a hole a consumer might forward-fill).
      - an UNKNOWN-session fraction above PANEL_MAX_BROKEN_SESSION_FRACTION,
        where unknown = broken UNION omitted (see below).
      - for a path-dependent indicator, ANY unknown session inside the span.
      - a ``parameters`` or ``provenance`` value JSON cannot represent, refused
        by name rather than by a mid-write TypeError.

    ``seasoning_window`` and ``path_dependent`` are required rather than
    inferred from ``parameters``: the seasoning lookback is not recoverable from
    an indicator's parameter dict (a 19/39 EMA oscillator seasons on a 2-bar
    advance/decline lookback, not on 19 or 39), and guessing it would silently
    change which sessions count as broken.

    THE TWO WAYS A SESSION GOES MISSING, and why the ceiling is applied to their
    union. ``coverage_report``'s ``broken`` column is narrow on purpose: a
    SEASONED member with no print. A session can also simply be ABSENT from the
    series -- fewer than ``min_members`` resolved, or an indicator's own lookback
    (net-advance needs a previous close) leaving it undefined -- and the caller's
    ``.dropna()`` erases the distinction. Counting only ``broken`` would let a
    hole of about 300 sessions through with ``broken_session_fraction=0.0``
    (a unit test builds one through this exact API on a 900-session panel
    with one member leaving and another joining). Both lists are named
    individually in provenance and the ceiling is applied to their union.

    The FRACTIONS are defined explicitly so a future reader can reproduce them:
    denominator = the panel sessions from the snapshot's own first dated
    observation through the panel's last session; ``broken_session_fraction``'s
    numerator = the broken sessions inside that span;
    ``unknown_session_fraction``'s numerator = broken UNION omitted, and it is
    the one the ceiling gates on. The denominator runs to the PANEL's last
    session rather than the series', so a snapshot that simply stops early is an
    omission too -- ``as_of`` forward-fills past the last dated observation with
    exactly the same silence as it does across an interior hole. A caller who
    genuinely wants a shorter record should build a shorter panel.
    """
    if not isinstance(indicator, str) or not indicator:
        raise ValueError("write_breadth_snapshot: indicator must be non-empty")
    if not isinstance(parameters, dict):
        raise ValueError("write_breadth_snapshot: parameters must be a dict")
    if not isinstance(path_dependent, bool):
        raise ValueError("write_breadth_snapshot: path_dependent must be bool")
    seasoning_window = _check_window(seasoning_window, "seasoning_window")
    built_at_iso = _iso_utc(built_at)

    membership = panel.membership
    if membership.get("point_in_time") is not True or (
            membership.get("ledgerable") is not True):
        raise ValueError(
            "write_breadth_snapshot: refusing to write a snapshot on basis "
            f"{membership.get('basis_id')!r} -- point_in_time="
            f"{membership.get('point_in_time')!r}, ledgerable="
            f"{membership.get('ledgerable')!r}, reason="
            f"{membership.get('ledger_block_reason')!r}. A panel built from "
            "today's membership and run back through history measures a "
            "universe selected for having survived.")

    if not isinstance(series, pd.Series) or series.empty:
        raise ValueError(
            "write_breadth_snapshot: series must be a non-empty Series")
    index = pd.DatetimeIndex(series.index)
    if index.tz is not None:
        raise ValueError("write_breadth_snapshot: series index must be tz-naive")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError(
            "write_breadth_snapshot: series index must be sorted and unique")
    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            "write_breadth_snapshot: series contains NaN/inf. A broken or thin "
            "session must be DROPPED (call .dropna()), never persisted as a "
            "hole -- a consumer's forward-fill would turn 'unknown' into a "
            "fabricated observation.")
    off_calendar = index.difference(panel.sessions)
    if len(off_calendar) > 0:
        raise ValueError(
            "write_breadth_snapshot: series carries dates that are not panel "
            f"sessions: {[str(d.date()) for d in off_calendar[:5]]}")

    coverage = coverage_report(panel, window=seasoning_window)
    span = panel.sessions[panel.sessions >= index[0]]
    broken_in_span = coverage.loc[span, "broken"]
    broken = pd.DatetimeIndex(broken_in_span.index[broken_in_span.to_numpy()])
    # Panel sessions inside this snapshot's own span that carry NO observation.
    # The off-calendar check above validates only the converse (every SERIES
    # date must be a panel session), so without this a panel session the
    # caller's .dropna() removed for any reason other than a narrow `broken`
    # would be invisible to both the ceiling and the provenance record.
    omitted = span.difference(index)
    unknown = broken.union(omitted)

    broken_dates = [str(stamp.date()) for stamp in broken]
    omitted_dates = [str(stamp.date()) for stamp in omitted]
    unknown_dates = [str(stamp.date()) for stamp in unknown]
    fraction = float(len(broken)) / float(len(span))
    unknown_fraction = float(len(unknown)) / float(len(span))

    if path_dependent and unknown_dates:
        raise ValueError(
            f"write_breadth_snapshot: indicator {indicator!r} is path-dependent "
            f"and the span contains {len(unknown_dates)} unknown session(s) "
            f"{unknown_dates[:25]} ({len(broken_dates)} broken, "
            f"{len(omitted_dates)} absent from the series). .cumsum()/.ewm() "
            "skip NaN and would bridge the gap invisibly.")
    if unknown_fraction > constants.PANEL_MAX_BROKEN_SESSION_FRACTION:
        raise ValueError(
            f"write_breadth_snapshot: unknown-session fraction "
            f"{unknown_fraction!r} ({len(unknown_dates)} of {len(span)} "
            f"sessions: {len(broken_dates)} broken, {len(omitted_dates)} absent "
            f"from the series) exceeds PANEL_MAX_BROKEN_SESSION_FRACTION="
            f"{constants.PANEL_MAX_BROKEN_SESSION_FRACTION}. Unknown dates: "
            f"{unknown_dates[:25]}. Re-fetch the missing bars into a new dated "
            "cache directory rather than lowering the bar.")

    sizes = coverage.loc[index, "resolved_n"]
    denominator_by_size = {
        str(int(size)): int(count)
        for size, count in sorted(sizes.value_counts().items())}

    dates = [str(stamp.date()) for stamp in index]
    value_list = [float(v) for v in values]

    provenance = dict(panel.provenance)
    provenance.update({
        "basis_id": membership.get("basis_id"),
        "point_in_time": True,
        "ledgerable": True,
        "ledger_block_reason": None,
        "seasoning_rule": SEASONING_RULE,
        "seasoning_window": seasoning_window,
        "panel_first_session": str(panel.sessions[0].date()),
        "panel_last_session": str(panel.sessions[-1].date()),
        "denominator_by_size": denominator_by_size,
        "broken_sessions": broken_dates,
        "broken_session_fraction": fraction,
        "omitted_sessions": omitted_dates,
        "unknown_session_fraction": unknown_fraction,
        "path_dependent": path_dependent,
        "built_at": built_at_iso,
    })

    # Both dicts reach disk AND the content hash, so both must be provably
    # serializable before anything is opened.
    _require_json_serializable(parameters, "write_breadth_snapshot: parameters")
    _require_json_serializable(provenance, "write_breadth_snapshot: provenance")

    content_sha256 = _canonical_sha256(
        schema_version=SNAPSHOT_SCHEMA_VERSION, indicator=indicator,
        parameters=parameters, dates=dates, values=value_list,
        provenance=provenance)

    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "indicator": indicator,
        "parameters": parameters,
        "dates": dates,
        "values": value_list,
        "content_sha256": content_sha256,
        "provenance": provenance,
    }
    # Serialize COMPLETELY before touching the filesystem: json.dump writes
    # incrementally and raises mid-stream, which with a direct
    # `open(path, 'w')` would leave a truncated file at the destination.
    text = json.dumps(payload, indent=2, sort_keys=False, allow_nan=False)

    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    descriptor, temp_path = tempfile.mkstemp(
        prefix=".snapshot-", suffix=".json.tmp", dir=parent or None)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temp_path, path)
    except BaseException:
        # The destination keeps whatever bytes it already had; only the temp
        # file is discarded. KeyboardInterrupt is caught deliberately -- an
        # interrupted write must not leave litter next to a ledgered artifact.
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    return {"path": path, "content_sha256": content_sha256}


def load_breadth_snapshot(path: str, *, expect_sha256: str) -> AsOfSeries:
    """Load a snapshot as a lookup-only handle. Three unconditional refusals.

    1. ``provenance.point_in_time`` is not True, or ``ledgerable`` is not True.
       There is deliberately NO keyword that suppresses this: the contaminated,
       survivorship-biased basis cannot be loaded by any invocation.
    2. The stored ``content_sha256`` does not match the content it labels --
       and "the content it labels" is EVERYTHING the file asserts: schema
       version, indicator, parameters, dates, values AND the whole provenance
       dict. With provenance outside the hash, a snapshot whose
       ``broken_sessions`` were rewritten to ``['FORGED']`` would load against
       its original pin without complaint.
    3. The stored sha does not match ``expect_sha256`` -- which is keyword-only
       and has NO default. This is what stops a REBUILT snapshot from silently
       changing what an already-pinned family's holdout re-run measures: the
       catalog entry names the exact bytes, and a rebuild produces a loud
       failure on the next run rather than a quiet re-measurement.
    """
    payload = _read_json(path, "load_breadth_snapshot")
    for key in ("schema_version", "indicator", "parameters", "dates", "values",
                "content_sha256", "provenance"):
        if key not in payload:
            raise ValueError(
                f"load_breadth_snapshot: {path!r} is missing key {key!r}")
    provenance = payload["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("load_breadth_snapshot: provenance must be an object")

    if provenance.get("point_in_time") is not True or (
            provenance.get("ledgerable") is not True):
        raise ValueError(
            f"load_breadth_snapshot: refusing snapshot {path!r} on basis "
            f"{provenance.get('basis_id')!r} -- point_in_time="
            f"{provenance.get('point_in_time')!r}, ledgerable="
            f"{provenance.get('ledgerable')!r}, reason="
            f"{provenance.get('ledger_block_reason')!r}. A candidate scored on "
            "a survivor-biased panel must not be recordable as a real trial.")

    # Checked BEFORE the hash so a future format fails with a version message
    # rather than an inscrutable sha mismatch -- schema_version is inside the
    # hashed payload, so any change to it also reddens the check below.
    if payload["schema_version"] != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"load_breadth_snapshot: {path!r} has schema_version "
            f"{payload['schema_version']!r}, but this module writes and reads "
            f"version {SNAPSHOT_SCHEMA_VERSION}. A snapshot from a different "
            "format is not evidence for this one.")

    recomputed = _canonical_sha256(
        schema_version=payload["schema_version"], indicator=payload["indicator"],
        parameters=payload["parameters"], dates=payload["dates"],
        values=payload["values"], provenance=provenance)
    stored = payload["content_sha256"]
    if recomputed != stored:
        raise ValueError(
            f"load_breadth_snapshot: {path!r} content sha256 {recomputed} does "
            f"not match its own stored {stored}; the file was edited after it "
            "was written")
    if stored != expect_sha256:
        raise ValueError(
            f"load_breadth_snapshot: {path!r} has content sha256 {stored} but "
            f"the caller pinned {expect_sha256}. A rebuilt snapshot changes "
            "what an already-pinned family's holdout re-run measures; re-pin "
            "deliberately or restore the pinned bytes.")

    return AsOfSeries(
        dates=pd.DatetimeIndex(pd.to_datetime(payload["dates"])),
        values=payload["values"], indicator=payload["indicator"],
        content_sha256=stored)
