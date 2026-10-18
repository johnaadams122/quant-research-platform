"""Build the sector_spdr_11 breadth panel FROM DISK: fetch -> seal -> snapshot.

``src/mining/panel.py`` already knows how to load a membership file, verify a
sealed price cache against its manifest, align the closes into a Panel, reduce
that Panel to a breadth series and persist the result as a sha-pinned snapshot.
What it deliberately does NOT do is touch the network or the wall clock. This
script is the thin, impure shell around it: it fetches, it stamps, and it
hands everything else to the library.

Three subcommands, one per artifact:

    fetch     fetch every member's full yfinance history into a NEW dated
              cache directory, write the manifest, and SEAL the directory.
    snapshot  load a sealed cache + the membership file, build the Panel,
              compute one breadth indicator, drop the unknown sessions, and
              write a snapshot whose provenance names every dropped date.
    verify    re-load a snapshot against a pinned sha and print a few as_of()
              lookups as a smoke check.

    py scripts/build_sector_panel.py fetch --cache-dir data/mining/panel/cache_sector_spdr_11_<YYYY-MM-DD>
    py scripts/build_sector_panel.py snapshot --cache-dir ... --membership ... \\
        --indicator participation --params '{"window": 200, "min_members": 9}' \\
        --out data/mining/panel/snapshots/<name>.json
    py scripts/build_sector_panel.py verify --snapshot <path> --sha <hex>

THE MANIFEST IS THE SEAL. ``panel.CACHE_SEAL_FILENAME`` is ``MANIFEST.json``
and this host's filesystem is case-insensitive, so a separate lowercase
``manifest.json`` would be the SAME file. Rather than fight that, the manifest
is written under the seal's own name, atomically, as the very last step of a
successful fetch. A directory therefore holds either zero attestation (and
``panel.load_cached_closes`` refuses it) or a complete one; there is no state
in which CSVs are attested but the directory is not sealed.

FAIL CLOSED ON FETCH. ``constants.PANEL_MAX_FETCH_FAILURES`` is zero. Any
member that fails to fetch -- no rows, an exception, a first bar LATER than
the membership file's recorded start -- aborts the whole fetch, removes every
CSV this run wrote, and writes no manifest, so nothing a loader would accept
is left behind.

NO WALL-CLOCK READS IN LIBRARY CODE. ``write_snapshot`` takes ``built_at`` as a
required tz-aware datetime; only the CLI's ``--built-at`` default reads the
clock, matching ``panel.write_breadth_snapshot``'s own idiom.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import constants  # noqa: E402
from mining import panel  # noqa: E402


BASIS_ID = "sector_spdr_11"
MEMBERSHIP_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
CSV_COLUMNS = ("Open", "High", "Low", "Close")

DEFAULT_MEMBERSHIP_PATH = os.path.join(
    _ROOT, "data", "mining", "panel", "sector_spdr_11_membership.json")

# The eleven Select Sector SPDR funds and their FIRST BAR dates as recorded in
# constants.py's PANEL_MIN_RESOLVED_MEMBERS comment (read off each fund's
# yfinance history). ``fetch`` re-verifies every one of these against the
# fund's own max-period history and aborts if a first bar is later than the
# recorded start; an earlier first bar is fine.
SECTOR_SPDR_11_MEMBERS = (
    ("XLB", "1998-12-22", None),
    ("XLE", "1998-12-22", None),
    ("XLF", "1998-12-22", None),
    ("XLI", "1998-12-22", None),
    ("XLK", "1998-12-22", None),
    ("XLP", "1998-12-22", None),
    ("XLU", "1998-12-22", None),
    ("XLV", "1998-12-22", None),
    ("XLY", "1998-12-22", None),
    ("XLRE", "2015-10-08", None),
    ("XLC", "2018-06-19", None),
)

# The Panel-taking reducers in panel.py, keyed by the CLI's indicator name.
#   required      parameter names the reducer needs besides min_members
#   seasoning     how the snapshot's seasoning_window is derived from params
#   path_dependent whether the reducer accumulates (cumsum / ewm) -- such a
#                 snapshot refuses ANY unknown session inside its span
# The seasoning rule per indicator follows the reducers' own docstrings:
# participation seasons on its own window; every advance/decline-based
# reducer seasons on panel._ADVANCE_DECLINE_LOOKBACK (2 bars), including the
# 19/39 oscillator, which write_breadth_snapshot's docstring calls out
# explicitly.
INDICATORS = {
    "participation": {
        "fn": panel.breadth_participation,
        "required": ("window",),
        "seasoning": lambda p: int(p["window"]),
        "path_dependent": False,
    },
    "net_advance_ratio": {
        "fn": panel.breadth_net_advance_ratio,
        "required": (),
        "seasoning": lambda p: panel._ADVANCE_DECLINE_LOOKBACK,
        "path_dependent": False,
    },
    "net_advance_oscillator": {
        "fn": panel.breadth_net_advance_oscillator,
        "required": ("fast", "slow"),
        "seasoning": lambda p: panel._ADVANCE_DECLINE_LOOKBACK,
        "path_dependent": True,
    },
    "cumulative_ad_trend": {
        "fn": panel.breadth_cumulative_ad_trend,
        "required": ("trend_days",),
        "seasoning": lambda p: panel._ADVANCE_DECLINE_LOOKBACK,
        "path_dependent": True,
    },
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str) -> str:
    with open(path, "rb") as handle:
        return sha256_bytes(handle.read())


def _write_json_atomic(path: str, payload: dict) -> None:
    """Serialize completely, then temp-file + os.replace (panel.py's idiom)."""
    text = json.dumps(payload, indent=2, sort_keys=False, allow_nan=False)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    descriptor, temp_path = tempfile.mkstemp(
        prefix=".manifest-", suffix=".json.tmp", dir=parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temp_path, path)
    except BaseException:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def manifest_path_for(cache_dir: str) -> str:
    """The manifest lives under the seal's own name (see module docstring)."""
    return os.path.join(cache_dir, panel.CACHE_SEAL_FILENAME)


def panel_fingerprint(manifest: dict) -> str:
    """sha256 over the manifest's per-ticker CSV sha256s, in sorted ticker order.

    Text hashed: one ``TICKER:sha256`` line per ticker, joined by newlines,
    tickers sorted. Two caches with byte-identical CSVs for the same tickers
    therefore share a fingerprint regardless of fetch date or directory name.
    """
    per_ticker = manifest["per_ticker"]
    lines = [f"{ticker}:{per_ticker[ticker]['sha256']}"
             for ticker in sorted(per_ticker)]
    return sha256_bytes("\n".join(lines).encode("utf-8"))


def parse_built_at(text: str) -> datetime:
    """Parse an ISO-8601 timestamp; it MUST carry a UTC offset."""
    candidate = text.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        stamp = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(
            f"--built-at {text!r} is not an ISO-8601 timestamp") from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError(
            f"--built-at {text!r} has no UTC offset; a snapshot's built_at "
            "must be timezone-aware (e.g. 2026-09-04T21:00:00Z)")
    return stamp


# ---------------------------------------------------------------------------
# Membership file
# ---------------------------------------------------------------------------

def build_membership_dict(members=SECTOR_SPDR_11_MEMBERS, *,
                          basis_id: str = BASIS_ID, notes: str = "") -> dict:
    """The exact dict shape panel.load_membership accepts, fully validated."""
    membership = {
        "basis_id": basis_id,
        "schema_version": MEMBERSHIP_SCHEMA_VERSION,
        "point_in_time": True,
        "ledgerable": True,
        "ledger_block_reason": None,
        "notes": notes,
        "members": [
            {"ticker": ticker, "start": start, "end": end}
            for ticker, start, end in members],
    }
    panel._validate_membership(membership)
    return membership


def write_membership_file(path: str, membership: dict) -> str:
    panel._validate_membership(membership)
    _write_json_atomic(path, membership)
    return path


# ---------------------------------------------------------------------------
# Fetch + cache + seal
# ---------------------------------------------------------------------------

def _default_fetcher(ticker: str) -> pd.DataFrame | None:
    """yfinance via data_pipeline.fetch_ohlc (auto_adjust=True, period=max).

    Imported lazily so the library-style functions above stay importable in a
    test process with no network and no yfinance import.
    """
    from data_pipeline import fetch_ohlc  # noqa: WPS433 -- deliberate
    return fetch_ohlc(ticker, start="max")


def normalize_fetched_frame(ticker: str, frame) -> pd.DataFrame:
    """Coerce a fetched frame into the CSV shape the loader reads back.

    Accepts data_pipeline.fetch_ohlc's output (a ``Date`` column of
    ``datetime.date`` objects) or any frame with a DatetimeIndex. Returns a
    frame indexed by tz-naive normalized ``Date`` with exactly CSV_COLUMNS,
    sorted unique, and with the same validation panel.load_cached_closes will
    apply to Close on the way back in -- so a frame that passes here cannot be
    refused later.
    """
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"fetch for {ticker!r} returned no rows")
    work = frame.copy()
    if "Date" in work.columns:
        work = work.set_index("Date")
    if not isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.DatetimeIndex(pd.to_datetime(work.index))
    if work.index.tz is not None:
        work.index = work.index.tz_localize(None)
    work.index = pd.DatetimeIndex(work.index).normalize()
    work.index.name = "Date"
    missing = [column for column in CSV_COLUMNS if column not in work.columns]
    if missing:
        raise ValueError(
            f"fetch for {ticker!r} is missing columns {missing} "
            f"(found {list(work.columns)})")
    work = work.loc[:, list(CSV_COLUMNS)].astype(float)
    work = work[~work.index.duplicated(keep="first")].sort_index()
    if work.index.has_duplicates or not work.index.is_monotonic_increasing:
        raise ValueError(f"fetch for {ticker!r}: index is not sorted unique")
    closes = work["Close"].to_numpy(dtype=float)
    if not np.isfinite(closes).all():
        raise ValueError(f"fetch for {ticker!r}: non-finite Close values")
    if (closes <= 0.0).any():
        raise ValueError(f"fetch for {ticker!r}: non-positive Close values")
    return work


def write_cache(cache_dir: str, tickers, *, fetcher=_default_fetcher,
                basis_id: str = BASIS_ID, membership: dict | None = None,
                membership_path: str | None = None,
                fetch_date: str | None = None) -> dict:
    """Fetch every ticker into ``cache_dir``, write the manifest, seal. Or abort.

    Refuses a sealed directory (panel.assert_cache_dir_unsealed) and a
    directory that already holds any CSV (an unattested leftover would be
    indistinguishable from this run's output). On ANY failure -- a fetch that
    returns nothing, raises, or produces a first bar later than the
    membership file's recorded start -- every CSV this run wrote is removed
    and no manifest is written. Returns the manifest dict on success.
    """
    tickers = sorted(dict.fromkeys(tickers))
    if not tickers:
        raise ValueError("write_cache: no tickers to fetch")
    panel.assert_cache_dir_unsealed(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    leftovers = sorted(name for name in os.listdir(cache_dir)
                       if name.lower().endswith(".csv"))
    if leftovers:
        raise ValueError(
            f"write_cache: {cache_dir!r} already holds CSVs {leftovers[:5]} "
            "but no seal -- an unattested leftover from an aborted fetch. "
            "Delete the directory or fetch into a new dated one.")

    starts = {}
    if membership is not None:
        panel._validate_membership(membership)
        starts = {entry["ticker"]: pd.Timestamp(entry["start"])
                  for entry in membership["members"]}

    written = []
    per_ticker = {}

    def _abort(message: str) -> None:
        for path in written:
            if os.path.exists(path):
                os.remove(path)
        raise RuntimeError(
            f"fetch ABORTED: {message}. PANEL_MAX_FETCH_FAILURES="
            f"{constants.PANEL_MAX_FETCH_FAILURES}; removed "
            f"{len(written)} partial CSV(s), wrote no manifest.")

    for ticker in tickers:
        try:
            frame = normalize_fetched_frame(ticker, fetcher(ticker))
        except Exception as exc:  # noqa: BLE001 -- every failure aborts
            _abort(f"member {ticker!r} failed to fetch ({exc})")
        first_date = frame.index[0]
        if ticker in starts and first_date > starts[ticker]:
            _abort(
                f"member {ticker!r} first bar {first_date.date()} is LATER "
                f"than its recorded membership start "
                f"{starts[ticker].date()}; the membership file's dates must "
                "be re-verified by a human before this basis is used")
        csv_path = os.path.join(cache_dir, f"{ticker}.csv")
        frame.to_csv(csv_path, lineterminator="\n")
        written.append(csv_path)
        per_ticker[ticker] = {
            "path": f"{ticker}.csv",
            "sha256": sha256_file(csv_path),
            "rows": int(len(frame)),
            "first_date": str(first_date.date()),
            "last_date": str(frame.index[-1].date()),
        }

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "basis_id": basis_id,
        "adjustment": constants.PANEL_PRICE_ADJUSTMENT,
        "fetch_date": fetch_date,
        "fetch_source": "yfinance via data_pipeline.fetch_ohlc(period=max, auto_adjust=True)",
        "membership_path": membership_path,
        "membership_sha256": (sha256_file(membership_path)
                              if membership_path else None),
        "tickers_requested": tickers,
        "tickers_cached": sorted(per_ticker),
        "failed": {},
        "per_ticker": per_ticker,
    }
    manifest["panel_fingerprint"] = panel_fingerprint(manifest)
    _write_json_atomic(manifest_path_for(cache_dir), manifest)
    return manifest


# ---------------------------------------------------------------------------
# Panel assembly
# ---------------------------------------------------------------------------

def build_sessions(closes: dict) -> pd.DatetimeIndex:
    """UNION of every member's own trading days: sorted, unique, tz-naive."""
    if not closes:
        raise ValueError("build_sessions: no price series supplied")
    union = None
    for ticker, series in closes.items():
        index = pd.DatetimeIndex(series.index)
        if index.tz is not None:
            raise ValueError(f"build_sessions: {ticker!r} index is tz-aware")
        union = index if union is None else union.union(index)
    sessions = pd.DatetimeIndex(union).normalize().unique().sort_values()
    if len(sessions) == 0:
        raise ValueError("build_sessions: empty union")
    return sessions


def build_sector_panel(*, cache_dir: str, membership_path: str,
                       manifest_path: str | None = None) -> panel.Panel:
    """Sealed cache + membership file -> Panel with a real provenance dict."""
    if manifest_path is None:
        manifest_path = manifest_path_for(cache_dir)
    membership = panel.load_membership(membership_path)
    manifest = panel.load_panel_manifest(manifest_path)
    if manifest["basis_id"] != membership["basis_id"]:
        raise ValueError(
            f"build_sector_panel: manifest basis {manifest['basis_id']!r} is "
            f"not the membership basis {membership['basis_id']!r}")
    tickers = sorted(entry["ticker"] for entry in membership["members"])
    closes = panel.load_cached_closes(
        cache_dir, tickers, manifest_path=manifest_path)
    sessions = build_sessions(closes)
    provenance = {
        "basis_id": membership["basis_id"],
        "membership_path": os.path.relpath(membership_path, _ROOT).replace(
            os.sep, "/"),
        "membership_sha256": sha256_file(membership_path),
        "manifest_path": os.path.relpath(manifest_path, _ROOT).replace(
            os.sep, "/"),
        "panel_fingerprint": panel_fingerprint(manifest),
        "adjustment": constants.PANEL_PRICE_ADJUSTMENT,
        "sessions_source": "union of member trading days from the sealed cache",
    }
    return panel.build_panel(
        prices_by_ticker=closes, membership=membership, sessions=sessions,
        provenance=provenance)


# ---------------------------------------------------------------------------
# Indicators + snapshots
# ---------------------------------------------------------------------------

def compute_indicator(panel_obj: panel.Panel, indicator: str,
                      params: dict | None = None) -> tuple:
    """Run one reducer, DROP unknown sessions (never fill).

    Returns ``(series, parameters, seasoning_window, path_dependent)`` where
    ``parameters`` is the fully-resolved dict (min_members defaulted to
    constants.PANEL_MIN_RESOLVED_MEMBERS) that the snapshot records.
    """
    if indicator not in INDICATORS:
        raise ValueError(
            f"unknown indicator {indicator!r}; choose one of "
            f"{sorted(INDICATORS)}")
    spec = INDICATORS[indicator]
    params = dict(params or {})
    allowed = set(spec["required"]) | {"min_members"}
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(
            f"indicator {indicator!r} does not take parameters {unknown}; "
            f"allowed: {sorted(allowed)}")
    missing = [name for name in spec["required"] if name not in params]
    if missing:
        raise ValueError(
            f"indicator {indicator!r} requires parameters {missing}")
    params.setdefault("min_members", constants.PANEL_MIN_RESOLVED_MEMBERS)
    for name, value in params.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(
                f"parameter {name!r} must be an int, got {value!r}")
    parameters = {name: int(params[name]) for name in sorted(params)}

    series = spec["fn"](panel_obj, **parameters)
    reduced = series.dropna()
    if reduced.empty:
        raise ValueError(
            f"indicator {indicator!r} produced no defined sessions at all "
            f"with parameters {parameters}")
    return (reduced, parameters, spec["seasoning"](parameters),
            spec["path_dependent"])


def write_snapshot(panel_obj: panel.Panel, indicator: str, params: dict | None,
                   out_path: str, *, built_at: datetime) -> dict:
    """compute_indicator + panel.write_breadth_snapshot, with a report dict."""
    reduced, parameters, seasoning_window, path_dependent = compute_indicator(
        panel_obj, indicator, params)
    result = panel.write_breadth_snapshot(
        reduced, out_path, indicator=indicator, panel=panel_obj,
        parameters=parameters, seasoning_window=seasoning_window,
        path_dependent=path_dependent, built_at=built_at)
    with open(out_path, "r", encoding="utf-8") as handle:
        provenance = json.load(handle)["provenance"]
    return {
        "path": os.path.relpath(out_path, _ROOT).replace(os.sep, "/"),
        "content_sha256": result["content_sha256"],
        "indicator": indicator,
        "parameters": parameters,
        "seasoning_window": seasoning_window,
        "path_dependent": path_dependent,
        "first_date": str(reduced.index[0].date()),
        "last_date": str(reduced.index[-1].date()),
        "n_sessions": int(len(reduced)),
        "panel_first_session": provenance["panel_first_session"],
        "panel_last_session": provenance["panel_last_session"],
        "panel_n_sessions": int(len(panel_obj.sessions)),
        "broken_sessions": provenance["broken_sessions"],
        "omitted_sessions": provenance["omitted_sessions"],
        "unknown_session_fraction": provenance["unknown_session_fraction"],
        "denominator_by_size": provenance["denominator_by_size"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_fetch(args) -> int:
    membership = panel.load_membership(args.membership)
    tickers = [entry["ticker"] for entry in membership["members"]]
    manifest = write_cache(
        args.cache_dir, tickers, basis_id=membership["basis_id"],
        membership=membership, membership_path=args.membership,
        fetch_date=args.fetch_date)
    summary = {
        "cache_dir": args.cache_dir,
        "manifest_path": manifest_path_for(args.cache_dir),
        "panel_fingerprint": manifest["panel_fingerprint"],
        "tickers_cached": manifest["tickers_cached"],
        "per_ticker": {
            ticker: {key: entry[key]
                     for key in ("rows", "first_date", "last_date")}
            for ticker, entry in manifest["per_ticker"].items()},
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def _cmd_snapshot(args) -> int:
    params = json.loads(args.params) if args.params else {}
    if not isinstance(params, dict):
        raise ValueError("--params must be a JSON object")
    built_at = (parse_built_at(args.built_at) if args.built_at
                else datetime.now(timezone.utc))
    built = build_sector_panel(
        cache_dir=args.cache_dir, membership_path=args.membership)
    report = write_snapshot(
        built, args.indicator, params, args.out, built_at=built_at)
    print(json.dumps(report, sort_keys=True))
    return 0


def _cmd_verify(args) -> int:
    series = panel.load_breadth_snapshot(args.snapshot, expect_sha256=args.sha)
    described = series.describe()
    first = pd.Timestamp(described["first_date"])
    last = pd.Timestamp(described["last_date"])
    probes = {
        "day_before_first": first - pd.Timedelta(days=1),
        "first": first,
        "mid": first + (last - first) / 2,
        "last": last,
        "day_after_last": last + pd.Timedelta(days=1),
    }
    described["as_of"] = {
        f"{label}={stamp.date()}": series.as_of(stamp)
        for label, stamp in probes.items()}
    print(json.dumps(described, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="fetch, attest and seal a new cache dir")
    fetch.add_argument("--cache-dir", required=True)
    fetch.add_argument("--membership", default=DEFAULT_MEMBERSHIP_PATH)
    fetch.add_argument("--fetch-date", default=None,
                       help="recorded in the manifest; defaults to today (UTC)")
    fetch.set_defaults(func=_cmd_fetch)

    snap = sub.add_parser("snapshot", help="build one breadth snapshot")
    snap.add_argument("--cache-dir", required=True)
    snap.add_argument("--membership", default=DEFAULT_MEMBERSHIP_PATH)
    snap.add_argument("--indicator", required=True, choices=sorted(INDICATORS))
    snap.add_argument("--params", default="{}", help="JSON object")
    snap.add_argument("--out", required=True)
    snap.add_argument("--built-at", default=None,
                      help="ISO-8601 with offset; defaults to now (UTC)")
    snap.set_defaults(func=_cmd_snapshot)

    verify = sub.add_parser("verify", help="re-load a snapshot against a sha")
    verify.add_argument("--snapshot", required=True)
    verify.add_argument("--sha", required=True)
    verify.set_defaults(func=_cmd_verify)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fetch" and args.fetch_date is None:
        args.fetch_date = str(datetime.now(timezone.utc).date())
    try:
        return args.func(args)
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
