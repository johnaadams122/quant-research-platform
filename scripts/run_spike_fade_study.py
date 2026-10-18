"""Driver of the range-bound spike-fade study, Stage 1.

WHAT IT DOES, once, end to end, with every definition frozen in advance in the
study's core modules (`mining.spike_fade_study`, `spike_fade_null`,
`spike_fade_analysis`; nothing here chooses a parameter, grid, seed, cutoff or rule):

  (a) data acquisition: yfinance daily history (period max, auto_adjust True)
      for every ticker of the study's trigger series and vehicles, and the FRED public daily CSVs
      (DCOILWTICO, DCOILBRENTEU, DHHNGSP, DEXJPUS; a '.' value is missing),
      fetched STRICTLY SERIALLY with a 3 second pause and ONE retry after 60
      seconds; the raw bytes are cached under <out>/data/raw/ with a manifest of
      SHA-256 digests and retrieval times (--skip-fetch reuses that cache and
      refuses a file whose digest no longer matches);
  (b) every series goes through the core's sealed loader (truncated at the
      cutoff BEFORE anything is computed) and data rules; the listing of the 20
      largest one-session moves, the JPY=X external cross-check against
      DEXJPUS, the GC=F / HG=F roll-gap check against GLD / CPER and the
      AR(1) half-life are produced per series;
  (c) the exploratory map: all 288 cells for every series and side, with the
      same-cell N1 null 90th percentile (B_map) as context;
  (d) the confirmatory test exactly as the null module
      exposes it (B replications, frozen seeds), the plateau ranking and
      stability, the vehicle tests within the tradable-vehicle scope, the put screen
      and the ABI-1 transfer line;
  (e) outputs in <out>/: REPORT.md (tracked evidence), results.json (every
      headline number and count, deterministic), cells.csv (every cell; bulk),
      run_meta.json (timestamps, workers, command: NOT deterministic) and the
      raw-data cache <out>/data/raw/ (bulk; do not track).

FAIL CLOSED. If any required series cannot be obtained (or a JPY=X series
would be built without its DEXJPUS reference) the study does NOT run on a
partial universe: results.json says status failed_closed, REPORT.md lists every
missing series with the reason, and the exit code is 3. Exit codes: 0 done,
1 unexpected error, 2 refused arguments (a cutoff after 2023-06-30), 3 failed
closed.

SEALED WINDOW. `--cutoff` defaults to `spike_fade_study.DISCOVERY_CUTOFF` and any later
date is refused. Every frame is truncated at the cutoff the moment it is parsed;
nothing after it reaches a series, a baseline, a vehicle or the ABI-1 account.

USAGE (Windows, from the repository root):
  python scripts/run_spike_fade_study.py --out runs/spike-fade-study --B 500 --B-map 200 --workers 4
  ... --skip-fetch                     reuse the cached raw data
  ... --universe "^VIX,GC=F"           restrict the trigger series (tests)

No trade is placed, no ledger row is written, no trial slot is spent, and no
currency amount is written anywhere.
"""
import os

# Keep BLAS single-threaded per process BEFORE numpy loads: the work is many
# small arrays and oversubscribed threads only make the machine unresponsive.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import concurrent.futures
import dataclasses
import hashlib
import io
import json
import math
import multiprocessing
import platform
import re
import sys
import time
import traceback
from collections import Counter, namedtuple
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mining import spike_fade_analysis as sfa  # noqa: E402
from mining import spike_fade_null as sfn  # noqa: E402
from mining import spike_fade_report as rep  # noqa: E402
from mining import spike_fade_study as sfs  # noqa: E402

DEFAULT_OUT = os.path.join(REPO_ROOT, "runs", "spike-fade-study")
SPEC_FILE = "spike-fade study pre-registration"  # label of the study definition this run implements (frozen in the core modules)
SPEC_VERSION = "3.1"
FETCH_PAUSE = 3.0
RETRY_WAIT = 60.0
MAX_WORKERS = 4
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
SCHEMA = "spike-fade-results-v1"
MIN_ROLL_GAP_OVERLAP = 30
ROLL_GAP_SD_MULTIPLE = 5.0
AR1_MIN_PAIRS = 30
TOP_HEADLINE_CELLS = 5
DATA_RULE_EXCLUDED = ("no_e1_confirmation", "tp_not_applicable")
DATA_LABELS = {
    "fred": ("current revised history (downloaded once on the run date; research-only, not "
             "on the repository's verified-unrevised allowlist)"),
    "yahoo": "vendor-adjusted history as retrieved on the run date",
}

Need = namedtuple("Need", "kind name source symbol")


@dataclasses.dataclass(frozen=True)
class SeriesSpec:
    """One trigger series of the study: its name in the study, where it
    comes from, the first usable date of the table (None = its own start), its
    role, the extra FRED publication lag and (for JPY=X only) the
    external same-day reference series."""
    name: str
    source: str
    symbol: str
    first_date: Optional[str]
    role: str
    extra_lag: int = 0
    reference: Optional[tuple] = None


UNIVERSE = (
    SeriesSpec("^VIX", "yahoo", "^VIX", "1990-01-02", "confirmatory"),
    SeriesSpec("WTI spot", "fred", "DCOILWTICO", "1986-01-02", "confirmatory", 1),
    SeriesSpec("Brent spot", "fred", "DCOILBRENTEU", "1987-05-20", "confirmatory", 1),
    SeriesSpec("Henry Hub spot", "fred", "DHHNGSP", "1997-01-07", "confirmatory", 1),
    SeriesSpec("GC=F", "yahoo", "GC=F", "2000-08-30", "confirmatory"),
    SeriesSpec("HG=F", "yahoo", "HG=F", "2000-08-30", "confirmatory"),
    SeriesSpec("DX-Y.NYB", "yahoo", "DX-Y.NYB", "1990-01-02", "confirmatory"),
    SeriesSpec("^OVX", "yahoo", "^OVX", None, "exploratory"),
    SeriesSpec("^VVIX", "yahoo", "^VVIX", None, "exploratory"),
    SeriesSpec("^GVZ", "yahoo", "^GVZ", None, "exploratory"),
    SeriesSpec("^MOVE", "yahoo", "^MOVE", None, "exploratory"),
    SeriesSpec("^TNX", "yahoo", "^TNX", "1990-01-02", "exploratory"),
    SeriesSpec("JPY=X", "yahoo", "JPY=X", "1990-01-02", "exploratory",
               reference=("fred", "DEXJPUS")),
)
ROLL_GAP_PAIRS = (("GC=F", "GLD"), ("HG=F", "CPER"))
CORE_SYMBOL = "SPY"


class CutoffRefused(Exception):
    """A cutoff after the discovery cutoff was requested."""


class CacheError(Exception):
    """A cached raw file is missing, unlisted or fails its SHA-256 digest."""


class MissingReference(Exception):
    """A series that needs an external reference (JPY=X against DEXJPUS) was
    about to be built without it."""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _default_log(msg: str) -> None:
    """Timestamped log line; bare runs of 8 to 17 digits (the frozen seeds, which
    the null module prints in its own report lines) are grouped with underscores
    (2026_0923) so no log line carries a long unbroken digit run."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {rep.group_long_digits(str(msg))}",
          flush=True)


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def json_safe(obj):
    """Recursively make a structure strict-JSON safe: NaN and infinities become
    None, numpy scalars and arrays become Python values, timestamps become
    ISO date strings, mapping keys become strings (a tuple key is joined with
    '|'), dataclasses become dicts. Unknown types raise TypeError (a silent
    str() would hide a bug)."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if obj is pd.NaT:
        return None
    if isinstance(obj, (pd.Timestamp, datetime)):
        ts = pd.Timestamp(obj)
        return ts.strftime("%Y-%m-%d") if (ts == ts.normalize()) else ts.isoformat()
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key = "|".join(str(p) for p in k) if isinstance(k, tuple) else str(k)
            out[key] = json_safe(v)
        return out
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in (sorted(obj) if isinstance(obj, (set, frozenset)) else obj)]
    if isinstance(obj, (np.ndarray, pd.Index, pd.Series)):
        return [json_safe(v) for v in obj.tolist()] if isinstance(obj, np.ndarray) else \
            [json_safe(v) for v in list(obj)]
    if isinstance(obj, pd.DataFrame):
        return [json_safe(r) for r in obj.to_dict(orient="records")]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return json_safe(dataclasses.asdict(obj))
    raise TypeError(f"cannot make {type(obj).__name__} JSON safe")


def write_json(path: str, payload) -> None:
    text = json.dumps(json_safe(payload), indent=1, allow_nan=False)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")


# ---------------------------------------------------------------------------
# The cutoff and the seal
# ---------------------------------------------------------------------------

def check_cutoff(cutoff) -> pd.Timestamp:
    """The discovery cutoff or an earlier date; a later one is refused."""
    ts = pd.Timestamp(cutoff).normalize()
    if ts > sfs.DISCOVERY_CUTOFF:
        raise CutoffRefused(
            f"cutoff {ts.date()} is after the sealed discovery cutoff "
            f"{sfs.DISCOVERY_CUTOFF.date()}; this study refuses to load later data")
    return ts


def seal_frame(df: pd.DataFrame, cutoff) -> tuple:
    """Truncate a parsed frame at the cutoff (inclusive) and count what the
    seal removed. Called the moment a frame is parsed, so nothing after the
    cutoff reaches any computation. Returns (frame, n_dropped_after_cutoff)."""
    cutoff = pd.Timestamp(cutoff)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    idx = idx.normalize()
    keep = np.asarray(idx <= cutoff)
    out = df.loc[keep].copy()
    out.index = idx[keep]
    if len(out) and out.index.max() > cutoff:
        raise sfs.SealedWindowError("seal_frame produced a date after the cutoff")
    return out, int((~keep).sum())


# ---------------------------------------------------------------------------
# Universe and requirements
# ---------------------------------------------------------------------------

def resolve_universe(arg) -> list:
    """The trigger series to run, in the canonical order of the study's series table
    (whatever order `arg` lists them in). None or an empty string is the full
    universe; an unknown name raises ValueError."""
    if arg is None or (isinstance(arg, str) and not arg.strip()):
        return list(UNIVERSE)
    names = arg.split(",") if isinstance(arg, str) else list(arg)
    by = {s.name: s for s in UNIVERSE}
    chosen = set()
    for raw in names:
        nm = raw.strip()
        if not nm:
            continue
        nm = nm if nm in by else sfa.canonical_series(nm)
        if nm not in by:
            raise ValueError(f"unknown series {raw!r}; known: {', '.join(by)}")
        chosen.add(nm)
    return [s for s in UNIVERSE if s.name in chosen]


def required_needs(specs) -> list:
    """Every download the given trigger series need: the series themselves,
    their external reference, the tradable vehicles of their pairs and the SPY
    core. No duplicates, fixed order."""
    needs, seen = [], set()

    def add(kind, name, source, symbol):
        if (source, symbol) in seen:
            return
        seen.add((source, symbol))
        needs.append(Need(kind, name, source, symbol))

    for spec in specs:
        add("trigger", spec.name, spec.source, spec.symbol)
        if spec.reference is not None:
            add("reference", spec.reference[1], *spec.reference)
    for spec in specs:
        for side in sfs.SIDES:
            v = sfa.VEHICLE_TABLE.get((spec.name, side))
            if v is not None:
                add("vehicle", v.ticker, "yahoo", v.ticker)
    add("core", CORE_SYMBOL, "yahoo", CORE_SYMBOL)
    return needs


# ---------------------------------------------------------------------------
# Raw files: parsing, fetching, the manifest, the cache
# ---------------------------------------------------------------------------

def raw_filename(source: str, symbol: str) -> str:
    return f"{source}_{re.sub(r'[^A-Za-z0-9._-]', '_', symbol)}.csv"


def yahoo_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """The canonical raw form of a yfinance history frame: date-only index, the
    OHLC and volume columns, ISO dates."""
    df = df.copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "Date"
    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    return df[cols].to_csv(date_format="%Y-%m-%d").encode("utf-8")


def parse_raw(source: str, raw: bytes) -> pd.DataFrame:
    """Parse cached raw bytes. Yahoo: a frame with at least a Close column.
    FRED: a one-column frame `value` (a '.' is NaN). Anything else raises
    ValueError (so an HTML error page counts as a failed download)."""
    try:
        if source == "yahoo":
            df = pd.read_csv(io.BytesIO(raw), index_col=0)
            if "Close" not in df.columns:
                raise ValueError("no Close column")
            df = df.apply(pd.to_numeric, errors="raise")
            idx = pd.to_datetime(df.index, errors="raise")
        elif source == "fred":
            df = pd.read_csv(io.BytesIO(raw), na_values=["."])
            if df.shape[1] < 2:
                raise ValueError("expected a date column and a value column")
            idx = pd.to_datetime(df.iloc[:, 0], errors="raise")
            df = pd.DataFrame({"value": pd.to_numeric(df.iloc[:, 1], errors="raise").to_numpy()})
        else:
            raise ValueError(f"unknown source {source!r}")
    except ValueError:
        raise
    except Exception as exc:                            # malformed bytes of any kind
        raise ValueError(f"unparseable {source} file: {exc}") from exc
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "Date"
    if len(df) == 0:
        raise ValueError(f"{source} file has no rows")
    return df


def real_fetcher(source: str, symbol: str) -> bytes:
    """The real network layer (used only by the real run)."""
    if source == "yahoo":
        import yfinance as yf
        raw = yf.Ticker(symbol).history(period="max", auto_adjust=True)
        if raw is None or raw.empty:
            raise RuntimeError(f"yfinance returned no rows for {symbol}")
        return yahoo_to_csv_bytes(raw)
    if source == "fred":
        import requests
        resp = requests.get(FRED_URL, params={"id": symbol}, timeout=60)
        resp.raise_for_status()
        return resp.content
    raise ValueError(f"unknown source {source!r}")


def _attempt(fetcher, need):
    try:
        raw = fetcher(need.source, need.symbol)
        if not isinstance(raw, (bytes, bytearray)):
            raise TypeError("the fetch layer must return bytes")
        df = parse_raw(need.source, bytes(raw))
        return bytes(raw), df, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def fetch_all(needs, fetcher, raw_dir: str, *, sleep=time.sleep, pause: float = FETCH_PAUSE,
              retry_wait: float = RETRY_WAIT, log=None) -> tuple:
    """Download every need STRICTLY SERIALLY: `pause` seconds before every fetch
    except the first, and on a failure (an exception, or bytes that do not
    parse) ONE retry after `retry_wait` seconds. A second failure is recorded
    and the loop moves on, so every missing series is reported at once. Each
    good payload is written unchanged to <raw_dir>/<file> and gets a manifest
    entry with its SHA-256 and UTC retrieval time. Returns (files, failures):
    files = {filename: entry}, failures = {(source, symbol): reason}."""
    log = log or (lambda m: None)
    os.makedirs(raw_dir, exist_ok=True)
    files, failures = {}, {}
    first = True
    for need in needs:
        if not first:
            sleep(pause)
        first = False
        raw, df, err = _attempt(fetcher, need)
        if raw is None:
            log(f"fetch of {need.source}:{need.symbol} failed ({err}); retrying once "
                f"after {retry_wait:g} seconds")
            sleep(retry_wait)
            raw, df, err = _attempt(fetcher, need)
        if raw is None:
            failures[(need.source, need.symbol)] = (
                f"download failed twice (one retry {retry_wait:g} seconds later): {err}")
            log(f"FAILED {need.source}:{need.symbol}: {err}")
            continue
        name = raw_filename(need.source, need.symbol)
        with open(os.path.join(raw_dir, name), "wb") as f:
            f.write(raw)
        files[name] = {
            "kind": need.kind, "name": need.name, "source": need.source,
            "symbol": need.symbol, "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw), "rows": int(len(df)),
            "first_date": df.index.min().strftime("%Y-%m-%d"),
            "last_date": df.index.max().strftime("%Y-%m-%d"), "retrieved_utc": _utcnow()}
        log(f"fetched {need.source}:{need.symbol}: {len(df)} rows, "
            f"{files[name]['first_date']} to {files[name]['last_date']}")
    return files, failures


def load_manifest(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("files", {})


def write_manifest(path: str, files: dict) -> None:
    write_json(path, {"generated_utc": _utcnow(),
                      "note": "SHA-256 digests and retrieval times of the raw files in this folder",
                      "files": files})


def load_cached(raw_dir: str, files: dict, need) -> bytes:
    """The cached bytes of a need, verified against the manifest digest.
    Raises CacheError for an unlisted, missing or altered file."""
    name = raw_filename(need.source, need.symbol)
    ent = files.get(name)
    if ent is None:
        raise CacheError(f"{name}: no manifest entry")
    path = os.path.join(raw_dir, name)
    if not os.path.exists(path):
        raise CacheError(f"{name}: cached file is missing")
    with open(path, "rb") as f:
        raw = f.read()
    got = hashlib.sha256(raw).hexdigest()
    if got != ent["sha256"]:
        raise CacheError(f"{name}: SHA-256 digest mismatch (manifest {ent['sha256'][:12]}..., "
                         f"file {got[:12]}...); the cached file was altered")
    return raw


# ---------------------------------------------------------------------------
# Building series
# ---------------------------------------------------------------------------

def build_trigger_series(spec: SeriesSpec, frames: dict, cutoff, n_post=None):
    """A trigger series through the core's sealed loader and data rules
    (sealed loader and data rules): its own first usable date, the extra FRED lag,
    and, for JPY=X, the external same-day cross-check against DEXJPUS, without
    which the series is REFUSED (MissingReference). `frames` maps (source,
    symbol) to a frame already truncated by `seal_frame`; `n_post` optionally
    maps (source, symbol) to the number of rows that seal removed, so the
    series can report them."""
    key = (spec.source, spec.symbol)
    frame = frames[key]
    close = frame["Close"] if spec.source == "yahoo" else frame["value"]
    ref = None
    if spec.reference is not None:
        if spec.reference not in frames:
            raise MissingReference(
                f"{spec.name} requires the {spec.reference[1]} external same-day reference check; "
                "refusing to build it without one")
        ref = frames[spec.reference]["value"]
    first = pd.Timestamp(spec.first_date) if spec.first_date else None
    s = sfs.build_series(close, name=spec.name, cutoff=cutoff, first_date=first, reference=ref,
                         extra_lag=spec.extra_lag)
    if n_post is not None:
        s = dataclasses.replace(s, n_post_cutoff_dropped=int(n_post.get(key, 0)))
    if len(s.dates) and s.dates.max() > pd.Timestamp(cutoff):
        raise sfs.SealedWindowError(f"{spec.name}: a loaded date is after the cutoff")
    return s


def build_vehicle_series(ticker: str, frames: dict, cutoff, n_post=None):
    frame = frames[("yahoo", ticker)]
    s = sfs.build_series(frame["Close"], name=ticker, cutoff=cutoff)
    if n_post is not None:
        s = dataclasses.replace(s, n_post_cutoff_dropped=int(n_post.get(("yahoo", ticker), 0)))
    return s


# ---------------------------------------------------------------------------
# Descriptive checks
# ---------------------------------------------------------------------------

def ar1_half_life(p) -> dict:
    """Descriptive AR(1) fit of the log level (OLS with intercept over the valid
    consecutive pairs): phi and the half-life ln(0.5)/ln(phi) in sessions when
    0 < phi < 1, else None with a note. Not a gate."""
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p) & (p > 0)
    pair = ok[:-1] & ok[1:]
    x = np.log(p[:-1][pair])
    y = np.log(p[1:][pair])
    n = int(len(x))
    if n < AR1_MIN_PAIRS:
        return {"phi": None, "half_life_sessions": None, "n_pairs": n,
                "note": "too few valid consecutive pairs"}
    xm, ym = x.mean(), y.mean()
    var = float(((x - xm) ** 2).sum())
    if var <= 0:
        return {"phi": None, "half_life_sessions": None, "n_pairs": n,
                "note": "constant series"}
    phi = float(((x - xm) * (y - ym)).sum() / var)
    if 0.0 < phi < 1.0:
        return {"phi": phi, "half_life_sessions": math.log(0.5) / math.log(phi), "n_pairs": n,
                "note": ""}
    note = ("phi at or above 1: no mean reversion detectable at this frequency" if phi >= 1.0
            else "phi at or below 0: no persistence")
    return {"phi": phi, "half_life_sessions": None, "n_pairs": n, "note": note}


def roll_gap_diagnostic(fut, etf) -> dict:
    """Data-quality check: the daily log-return correlation and the largest absolute
    difference between a continuous-futures series and its ETF over their
    overlap (returns between consecutive COMMON valid sessions, so both cover the
    same interval), flagged when that largest gap exceeds five times the
    futures series' own daily standard deviation over the overlap."""
    f = pd.Series(np.asarray(fut.p, dtype=float), index=fut.dates)
    e = pd.Series(np.asarray(etf.p, dtype=float), index=etf.dates)
    j = pd.concat([f, e], axis=1, join="inner").dropna()
    base = {"series": fut.name, "etf": etf.name, "n": 0, "corr": None, "max_abs_diff": None,
            "max_abs_diff_date": None, "sd_daily": None, "ratio": None, "flagged": None,
            "status": "insufficient_overlap"}
    if len(j) < MIN_ROLL_GAP_OVERLAP + 1:
        return base
    r = np.log(j).diff().dropna()
    rf, re_ = r.iloc[:, 0].to_numpy(), r.iloc[:, 1].to_numpy()
    diff = np.abs(rf - re_)
    k = int(np.argmax(diff))
    sd = float(np.std(rf, ddof=1))
    corr = float(np.corrcoef(rf, re_)[0, 1]) if np.std(rf) > 0 and np.std(re_) > 0 else None
    ratio = float(diff[k] / sd) if sd > 0 else None
    return {"series": fut.name, "etf": etf.name, "n": int(len(r)), "corr": corr,
            "max_abs_diff": float(diff[k]),
            "max_abs_diff_date": r.index[k].strftime("%Y-%m-%d"), "sd_daily": sd,
            "ratio": ratio,
            "flagged": bool(ratio is not None and ratio > ROLL_GAP_SD_MULTIPLE), "status": "ok"}


def vehicle_signal_dates(vehicle, trigger_dates, extra_lag: int = 0) -> pd.DatetimeIndex:
    """The kept anchors of a vehicle test on the vehicle's own calendar (the
    same steps `vehicle_cell_test` takes: last vehicle session on or before the
    trigger date plus the extra lag, uniform censoring on the vehicle
    calendar, duplicates merged). These are the signal bars of the ABI-1
    transfer line."""
    td = pd.DatetimeIndex(trigger_dates).normalize()
    va = sfs.map_anchors_to_vehicle(td, vehicle.dates, extra_lag)
    pos, _ = sfs.censor_episodes(va.positions, vehicle.n)
    return vehicle.dates[np.unique(pos)]


def cell_key(cell) -> str:
    entry, H, tp, sl = cell
    return f"{entry}|{H}|{tp}|{sfs.sl_label(sl)}"


def cell_of(entry, H, tp, sl_label) -> tuple:
    return (entry, int(H), tp, None if str(sl_label) in ("none", "None", "nan") else float(sl_label))


# ---------------------------------------------------------------------------
# The per-series map job (top level so a spawned worker can import it)
# ---------------------------------------------------------------------------

_STATUS_COLS = tuple(sfs.STATUS_NAMES.values()) + ("tp_not_applicable",)


def map_job(args) -> dict:
    """The exploratory map of ONE series: all 8 groups x 72 cells with their
    per-cell statistics, the same-cell N1 null 90th percentile (B_map) and the
    stability rule for every cell, plus every count the report
    needs. Pure function of (series, B_map): its result does not depend on
    which other series exist or on the worker that runs it."""
    series, B_map = args
    signals = sfs.compute_triggers(series)
    groups = {(t, s): sfs.evaluate_group(series, t, s, signals=signals)
              for t in sfs.TRIGGERS for s in sfs.SIDES}
    nulls = sfn.n1_cell_null_series(series, B=B_map, signals=signals)
    frames, ginfo, episodes = [], [], {}
    affected = {}
    for (trig, side), g in groups.items():
        df = g.to_frame()
        mn = nulls[(trig, side)]
        df["null_q90"] = np.asarray(mn.q90, dtype=float)
        df["null_n_failed"] = int(mn.n_failed)
        st = [sfa.stability_for_cell(g.results, c, series.dates) for c in g.results.cells]
        df["stab_passed"] = [bool(x.passed) for x in st]
        df["stab_a"] = [bool(x.a_pass) for x in st]
        df["stab_b"] = [None if x.b_pass is None else bool(x.b_pass) for x in st]
        df["stab_S_drop1"] = [x.S_drop1 for x in st]
        df["stab_S_drop2"] = [x.S_drop2 for x in st]
        df["stab_first_half_mean"] = [x.first_half_mean for x in st]
        df["stab_second_half_mean"] = [x.second_half_mean for x in st]
        frames.append(df)
        ginfo.append({"trigger": trig, "side": side,
                      "n_flagged_days": int(g.counts["n_flagged_days"]),
                      "n_episodes": int(g.counts["n_episodes"]),
                      "n_kept": int(g.counts["n_kept"]),
                      "dropped_censored": int(g.counts["dropped_censored"])})
        episodes[f"{trig}|{side}"] = {"t0": [int(v) for v in g.episodes.t0],
                                       "dates": [d.strftime("%Y-%m-%d") for d in g.episodes.dates]}
        rem = sfs.removed_observations(series, g.results)
        rem = rem[~rem["status"].isin(DATA_RULE_EXCLUDED)]
        for r in rem.itertuples():
            e = affected.setdefault(r.date.strftime("%Y-%m-%d"), {"statuses": set(), "n_cells": 0})
            e["statuses"].add(r.status)
            e["n_cells"] += 1
    frame = pd.concat(frames, ignore_index=True)
    totals = {c: int(frame[c].sum()) for c in _STATUS_COLS if c in frame.columns}
    diags = {t: sfs.trigger_diagnostics(series, signals[t]) for t in sfs.TRIGGERS}
    return {"name": series.name, "frame": frame, "groups": ginfo, "episodes": episodes,
            "cell_status_totals": totals, "trigger_diagnostics": diags,
            "affected_dates": [{"date": d, "statuses": sorted(v["statuses"]),
                                "n_cells": v["n_cells"]} for d, v in sorted(affected.items())]}


def run_jobs(fn, jobs, workers: int) -> list:
    """Run `fn` over `jobs` in order; in a spawned process pool when more than
    one worker is asked for (at most MAX_WORKERS), else in this process."""
    workers = max(1, min(int(workers), MAX_WORKERS, len(jobs) or 1))
    if workers == 1:
        return [fn(j) for j in jobs]
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        return list(ex.map(fn, jobs))


# ---------------------------------------------------------------------------
# Recommendation (conditions a to e) and plain-word reasons
# ---------------------------------------------------------------------------

_REASON_WORDS = {
    "headline_p_unavailable": "the one pre-planned yes-or-no test could not be run "
                              "(no group of events was big enough to test)",
    "headline_p_above_max": "the one pre-planned yes-or-no test was not convincing "
                            "(its p was above 0.10)",
    "vehicle_test_not_run": "the fund you could actually trade was not tested",
    "underpowered_on_vehicle": "the fund you could actually trade had too few trades "
                               "(fewer than 8) to judge",
    "vehicle_excess_not_positive": "the fund you could actually trade did not beat random "
                                   "entry dates",
    "null_percentile_below_min": "the fund you could actually trade did not clearly beat "
                                 "random entry dates (it beat fewer than 80% of them)",
}


def build_recommendation(candidate: dict, abi_tstar: Optional[dict], *,
                         tstar_series: Optional[str]) -> dict:
    """The decision table: Stage 2 (registering ONE fixed parameter set, a bespoke
    driver and a permanent scoring slot) is recommended only if (a) a candidate
    exists under the tradable-vehicle test, (b) the ABI-1 transfer line is at or above the
    in-sample bar and passes the drawdown refusals, (c) a decision is made on how
    the search is counted, (d) the T* pair's series is not contaminated (else
    the contamination must be explicitly accepted), and (e) a market-hours quote sample
    exists. The study can decide (a) and (b); (c), (d) when contaminated, and
    (e) stay open for the reader. `register_stage2` is True only when (a) and (b)
    hold, and then means 'consider it', never 'do it'."""
    a_met = bool(candidate and candidate.get("is_candidate"))
    a_detail = ("a candidate exists" if a_met
                else "; ".join(candidate.get("reasons") or ["not evaluated"]) if candidate
                else "not evaluated")
    b_met, b_detail = False, "the ABI-1 transfer line was not run"
    b_return_ok = b_dd_ok = None
    if abi_tstar:
        if abi_tstar.get("status") == "computed":
            f = abi_tstar["full"]
            nominal = float(f.get("in_sample_bar"))
            exact = f.get("dsr_bar_annualized")
            exact = float(exact) if exact is not None and math.isfinite(float(exact)) else None
            bar_used = max(nominal, exact) if exact is not None else nominal
            sharpe = float(f.get("excess_sharpe_annualized"))
            b_return_ok = bool(sharpe >= bar_used)
            b_dd_ok = bool(f.get("dd_passed"))
            b_met = bool(b_return_ok and b_dd_ok)
            b_detail = (f"annualized excess Sharpe {sharpe:.2f} against the nominal bar "
                        f"{nominal:.2f}"
                        + (f" and the exact bar for this account length {exact:.2f} (the larger "
                           "applies)" if exact is not None else "")
                        + f"; drawdown refusals {'passed' if b_dd_ok else 'not passed'}")
        else:
            b_detail = f"not computed: {abi_tstar.get('reason')}"
    contaminated = sfa.is_contaminated(tstar_series) if tstar_series else None
    if tstar_series is None:
        d_met, d_detail = None, "no T* pair: not evaluated"
    elif contaminated:
        d_met = None
        d_detail = (f"{tstar_series} is marked contaminated: confirmatory on the discovery window "
                    "only; Stage 2 additionally needs explicit acceptance of that")
    else:
        d_met, d_detail = True, f"{tstar_series} is not marked contaminated"
    conditions = [
        {"id": "a", "text": "a candidate exists under the tradable-vehicle test (headline p at most 0.10 and the "
                            "tradable fund beats random entries)", "met": a_met,
         "detail": a_detail},
        {"id": "b", "text": "the ABI-1 transfer line is at or above the in-sample bar and passes "
                            "the drawdown refusals", "met": b_met, "detail": b_detail},
        {"id": "c", "text": "a decision is made on how the search is counted (about 288 cells per "
                            "series were explored; the pipeline has no sanctioned "
                            "search-then-register procedure)", "met": None,
         "detail": "a judgment call, not decidable by the study"},
        {"id": "d", "text": "the T* pair's series is not contaminated (or that contamination is "
                            "explicitly accepted)", "met": d_met, "detail": d_detail},
        {"id": "e", "text": "a market-hours quote sample exists", "met": None,
         "detail": "not part of Stage 1; needed before any registration"},
    ]
    reasons = []
    if not a_met:
        for code in (candidate.get("reasons") if candidate else None) or ["vehicle_test_not_run"]:
            reasons.append(_REASON_WORDS.get(code, code.replace("_", " ")))
    if not b_met:
        if abi_tstar and abi_tstar.get("status") == "computed":
            if not b_return_ok:
                reasons.append("the simulated account (the market plus a quarter-size position "
                               "in the fund) did not earn enough extra return for the risk it "
                               "took (below the in-sample bar)")
            if not b_dd_ok:
                reasons.append("the simulated account's worst loss was too deep (deeper than the "
                               "market's own, or beyond the 35% limit)")
        else:
            reasons.append("the simulated account check could not be run for the best group")
    return {"register_stage2": bool(a_met and b_met), "conditions": conditions,
            "reasons_not": [] if (a_met and b_met) else reasons}


# ---------------------------------------------------------------------------
# Result-dict helpers
# ---------------------------------------------------------------------------

def _series_info(spec, series, mj, cells_df) -> dict:
    ext = [d.strftime("%Y-%m-%d") for d in series.external_dates]
    npos = [d.strftime("%Y-%m-%d") for d in series.nonpos_dates]
    lr = sfs.largest_abs_returns(series)
    return {
        "role": spec.role, "source": spec.source, "symbol": spec.symbol,
        "extra_lag": int(spec.extra_lag), "table_first_date": spec.first_date,
        "first_date": series.dates[0].strftime("%Y-%m-%d"),
        "last_date": series.dates[-1].strftime("%Y-%m-%d"), "n_rows": int(series.n),
        "counts": series.counts(),
        "external_reference": spec.reference[1] if spec.reference else None,
        "external_correction_dates": ext, "nonpositive_dates": npos,
        "largest_returns": [{"date": r.date.strftime("%Y-%m-%d"),
                             "prev_date": r.prev_date.strftime("%Y-%m-%d"),
                             "log_return": float(r.log_return),
                             "abs_log_return": float(r.abs_log_return)}
                            for r in lr.itertuples()],
        "ar1": ar1_half_life(series.p),
        "trigger_diagnostics": mj["trigger_diagnostics"], "groups": mj["groups"],
        "cell_status_totals": mj["cell_status_totals"], "affected_dates": mj["affected_dates"],
    }


_MAP_FIELDS = ("rank", "series", "side", "trigger", "entry", "H", "tp", "sl", "n", "mean_x", "t",
               "plateau_score", "n_eligible_neighbors", "hit_rate", "mean_pnl", "break_even_bps",
               "ci_lo", "ci_hi", "null_q90", "stab_passed", "stab_a", "stab_b")


def _row_dict(row) -> dict:
    d = {k: (row[k] if k in row else None) for k in _MAP_FIELDS}
    return json_safe(d)


def _vehicle_result_dict(r) -> dict:
    trades = r.trades.copy()
    return {
        "n_anchors_input": r.n_anchors_input, "dropped_no_session": r.dropped_no_session,
        "dropped_past_end": r.dropped_past_end, "dropped_short_history": r.dropped_short_history,
        "dropped_censored": r.dropped_censored, "dropped_duplicate": r.dropped_duplicate,
        "n_anchors_kept": r.n_anchors_kept, "n_kept_not_traded": r.n_kept_not_traded, "n": r.n,
        "mean_pnl_pp": 100.0 * r.mean_pnl, "hit_rate": r.hit_rate,
        "null_mean_pnl_pp": 100.0 * r.null_mean_pnl, "null_reps_used": r.null_reps_used,
        "null_reps_failed": r.null_reps_failed, "excess_pp": r.excess_pp,
        "excess_sigma": r.excess_sigma, "null_percentile": r.null_percentile,
        "null_percentile_x": r.null_percentile_x,
        "unconditional_mean_pnl_pp": 100.0 * r.unconditional_mean_pnl,
        "unconditional_mean_daily_return_pp": 100.0 * r.unconditional_mean_daily_return,
        "window_start": r.window_start, "window_end": r.window_end, "era": r.era,
        "underpowered_on_vehicle": bool(r.underpowered_on_vehicle),
        "contaminated": bool(r.contaminated), "B": r.B,
        "trades": trades.to_dict(orient="records") if len(trades) else [],
    }


def _account_dict(a) -> dict:
    return {
        "n_sessions": a.n_sessions, "n_trades": a.n_trades,
        "excess_sharpe_annualized": a.excess_sharpe_annualized, "in_sample_bar": a.in_sample_bar,
        "meets_in_sample_bar": a.meets_in_sample_bar, "dsr_bar_annualized": a.dsr_bar_annualized,
        "max_dd_account": a.max_dd_account, "max_dd_spy": a.max_dd_spy,
        "absolute_cap": a.absolute_cap, "dd_relative_passed": a.dd_relative_passed,
        "dd_absolute_passed": a.dd_absolute_passed, "dd_passed": a.dd_passed,
        "terminal_wealth_account": a.terminal_wealth_account,
        "terminal_wealth_spy": a.terminal_wealth_spy,
        "below_min_sessions": a.below_min_sessions, "risk_refusals": a.risk_refusals,
    }


def _frame_problems(vf: pd.DataFrame, sf: pd.DataFrame, limit: int = 8) -> str:
    """Name the dates that make an ABI-1 account unbuildable: a vehicle Open or
    Close that is not finite and positive, and sessions the vehicle and SPY do
    not share inside the vehicle's span. Empty string when there are none."""
    msgs = []
    vals = vf.to_numpy(dtype=float)
    bad = ~np.isfinite(vals).all(axis=1) | (vals <= 0).any(axis=1)
    if bad.any():
        ds = [d.strftime("%Y-%m-%d") for d in vf.index[bad]]
        msgs.append(f"the vehicle's Open or Close is not finite and positive on {len(ds)} "
                    f"date(s): {', '.join(ds[:limit])}" + (" ..." if len(ds) > limit else ""))
    span = sf.loc[vf.index[0]:vf.index[-1]].index
    only_vehicle = vf.index.difference(span)
    only_spy = span.difference(vf.index)
    for label, idx in (("vehicle sessions with no SPY bar", only_vehicle),
                       ("SPY sessions with no vehicle bar", only_spy)):
        if len(idx):
            ds = [d.strftime("%Y-%m-%d") for d in idx]
            msgs.append(f"calendar mismatch: {len(ds)} {label}: {', '.join(ds[:limit])}"
                        + (" ..." if len(ds) > limit else ""))
    return "; ".join(msgs)


def abi1_line(*, series_name, side, trigger, cell, ticker, position, signal_dates,
              vehicle_series, vehicle_frame, spy_frame, cutoff) -> dict:
    """One ABI-1 transfer line as a result dict. A ValueError from the account
    build (a calendar mismatch, a non-positive price, an account wiped out) is
    reported as `not_computed` with the reason, never patched around."""
    base = {"status": "not_computed", "reason": None, "series": series_name, "side": side,
            "trigger": trigger, "cell": cell_key(cell), "vehicle": ticker, "position": position}
    try:
        vf = vehicle_frame.loc[vehicle_series.dates, ["Open", "Close"]]
        lo = vf.index[0] - pd.Timedelta(days=10)
        sf = spy_frame.loc[lo:, ["Close"]]
        problems = _frame_problems(vf, sf)
        if problems:
            base["reason"] = problems
            return base
        tl = sfa.abi1_transfer_line(sf, vf, signal_dates, cell, position,
                                    vehicle_series=vehicle_series, cutoff=cutoff)
    except sfs.SealedWindowError:
        raise
    except Exception as exc:                         # reported with its reason, never patched around
        base["reason"] = f"{type(exc).__name__}: {exc}"
        return base
    base.update({
        "status": "computed", "n_signals": tl.n_signals,
        "n_skipped_overlap": tl.n_skipped_overlap, "n_skipped_no_entry": tl.n_skipped_no_entry,
        "n_skipped_undefined": tl.n_skipped_undefined, "full": _account_dict(tl.full),
        "ex_worst": _account_dict(tl.ex_worst), "conventions": tl.conventions,
        "overlay_notional": tl.overlay_notional, "cost_bps": tl.cost_bps,
        "trades": tl.trades.to_dict(orient="records") if len(tl.trades) else [],
        "dropped_trades": (tl.dropped_trades.to_dict(orient="records")
                           if len(tl.dropped_trades) else [])})
    return base


def _abi_summary(a: Optional[dict]) -> Optional[dict]:
    if a is None:
        return None
    if a.get("status") != "computed":
        return {"status": a.get("status"), "reason": a.get("reason")}
    f, x = a["full"], a["ex_worst"]
    return {"status": "computed", "excess_sharpe": f["excess_sharpe_annualized"],
            "bar": f["in_sample_bar"], "meets": f["meets_in_sample_bar"],
            "dd_passed": f["dd_passed"], "max_dd_account": f["max_dd_account"],
            "max_dd_spy": f["max_dd_spy"],
            "ex_worst_excess_sharpe": x["excess_sharpe_annualized"]}


def _put_dict(series_name, side, trigger, cell, role, res) -> dict:
    def screen(ps):
        if ps is None:
            return None
        out = {}
        for pct, row in ps.results.items():
            out[str(pct)] = {"break_even_premium": row["break_even_premium"],
                             "avg_premium": {str(k): v for k, v in row["avg_premium"].items()},
                             "avg_pnl": {str(k): v for k, v in row["avg_pnl"].items()},
                             "break_even_k": row["break_even_k"],
                             "break_even_k_status": row["break_even_k_status"]}
        return out
    obs = res["observed"]
    return {"series": series_name, "side": side, "trigger": trigger, "cell": cell_key(cell),
            "role": role, "H": obs.H, "status": "computed", "reason": None,
            "n_used": obs.n_used, "n_dropped": obs.n_dropped, "observed": screen(obs),
            "baseline": screen(res["baseline"]),
            "baseline_n_used": None if res["baseline"] is None else res["baseline"].n_used,
            "comparison": {str(k): v for k, v in (res["comparison"] or {}).items()}}


ROBUSTNESS_NAME = "null_size_robustness.json"


def _read_robustness(out_dir: str):
    """The synthetic null-size robustness result (<out>/calibration/
    null_size_robustness.json) or None if it is absent, unparseable or not
    finished. Read only by the report writer; no statistic depends on it."""
    path = os.path.join(out_dir, "calibration", ROBUSTNESS_NAME)
    try:
        with open(path, encoding="utf-8") as f:
            rob = json.load(f)
    except (OSError, ValueError):
        return None
    return rob if isinstance(rob, dict) and rob.get("finished") else None


def _spec_version_key(cal: dict) -> float:
    try:
        return float(cal.get("spec_version", 3.0))
    except (TypeError, ValueError):
        return 3.0


def _read_calibration(out_dir: str):
    """The synthetic-calibration result the report's power statement reads: among
    the finished JSON files in <out>/calibration/, the one for the NEWEST spec
    version (a file without a spec_version counts as version 3), ties broken by
    generation time then name. None if there is no finished, parseable file."""
    d = os.path.join(out_dir, "calibration")
    if not os.path.isdir(d):
        return None
    best, best_key = None, None
    for name in sorted(os.listdir(d)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, name), encoding="utf-8") as f:
                cal = json.load(f)
        except (OSError, ValueError):
            continue
        if not isinstance(cal, dict) or not cal.get("finished"):
            continue
        pw = cal.get("power")
        if not (isinstance(pw, dict) and pw.get("curve")):     # e.g. the size-robustness file
            continue
        key = (_spec_version_key(cal), str(cal.get("generated_utc", "")), name)
        if best_key is None or key > best_key:
            best, best_key = cal, key
    return best


# ---------------------------------------------------------------------------
# The study proper
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class StudyConfig:
    out: str
    B: int = sfn.NULL_B
    B_map: int = sfn.MAP_B
    workers: int = MAX_WORKERS
    skip_fetch: bool = False
    universe: Optional[str] = None
    cutoff: pd.Timestamp = sfs.DISCOVERY_CUTOFF
    fetch_pause: float = FETCH_PAUSE
    retry_wait: float = RETRY_WAIT
    argv: tuple = ()
    force: bool = False


def is_preregistered(cfg: StudyConfig, specs) -> tuple:
    """(True, []) only for the pre-registered configuration: the full 13-series
    universe, B = 500, B_map = 200 and the discovery cutoff. Anything else is a
    test of the machinery, and the report and results say so first."""
    notes = []
    if len(specs) != len(UNIVERSE):
        notes.append(f"restricted universe ({len(specs)} of {len(UNIVERSE)} series)")
    if int(cfg.B) != sfn.NULL_B:
        notes.append(f"B = {cfg.B} instead of {sfn.NULL_B}")
    if int(cfg.B_map) != sfn.MAP_B:
        notes.append(f"B_map = {cfg.B_map} instead of {sfn.MAP_B}")
    if pd.Timestamp(cfg.cutoff) != sfs.DISCOVERY_CUTOFF:
        notes.append(f"cutoff {pd.Timestamp(cfg.cutoff).date()} instead of "
                     f"{sfs.DISCOVERY_CUTOFF.date()}")
    return (not notes, notes)


def _params(cfg: StudyConfig, specs) -> dict:
    ok, notes = is_preregistered(cfg, specs)
    return {"pre_registered_configuration": ok, "configuration_notes": notes,
            "B": int(cfg.B), "B_map": int(cfg.B_map),
            "seeds": {"n1": sfn.N1_SEED, "n2": sfn.N2_SEED, "map": sfn.MAP_SEED,
                      "cluster_bootstrap": sfs.CLUSTER_BOOTSTRAP_SEED,
                      "vehicle_null": sfs.N1_SEED, "put_baseline": sfn.MAP_SEED},
            "universe": [s.name for s in specs],
            "confirmatory_family": [s.name for s in specs if s.role == "confirmatory"]}


def _command(cfg: StudyConfig, specs, *, cached: bool) -> str:
    try:
        rel = os.path.relpath(cfg.out, REPO_ROOT) if os.path.isabs(cfg.out) else cfg.out
    except ValueError:                               # another drive: keep the absolute path
        rel = cfg.out
    parts = [sys.executable.replace("\\", "/"), "scripts/run_spike_fade_study.py",
             "--out", rel.replace("\\", "/"), "--B", str(cfg.B), "--B-map", str(cfg.B_map),
             "--workers", str(cfg.workers)]
    if cfg.universe:
        parts += ["--universe", f'"{cfg.universe}"']
    if pd.Timestamp(cfg.cutoff) != sfs.DISCOVERY_CUTOFF:
        parts += ["--cutoff", pd.Timestamp(cfg.cutoff).strftime("%Y-%m-%d")]
    if cached or cfg.skip_fetch:
        parts.append("--skip-fetch")
    return " ".join(parts)


def _finish(cfg, specs, results, meta_extra, manifest_by_name, t_start) -> None:
    """Write results.json, run_meta.json and REPORT.md."""
    out = cfg.out
    calibration = _read_calibration(out)
    meta = {"command": _command(cfg, specs, cached=False),
            "command_to_reuse_cached_data": _command(cfg, specs, cached=True),
            "python": platform.python_version(), "workers": int(cfg.workers),
            "started_utc": t_start, "finished_utc": _utcnow(),
            "skip_fetch": bool(cfg.skip_fetch)}
    meta.update(meta_extra)
    write_json(os.path.join(out, "results.json"), results)
    write_json(os.path.join(out, "run_meta.json"), meta)
    text = rep.render_report(json_safe(results), meta=meta, calibration=calibration,
                             manifest=manifest_by_name, robustness=_read_robustness(out))
    with open(os.path.join(out, "REPORT.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _failed_results(cfg, specs, missing, reason, detail=()) -> dict:
    return {"schema": SCHEMA, "status": "failed_closed",
            "spec": {"file": SPEC_FILE, "version": SPEC_VERSION},
            "cutoff": pd.Timestamp(cfg.cutoff).strftime("%Y-%m-%d"),
            "params": _params(cfg, specs), "data": {"files": {}, "labels": DATA_LABELS},
            "failure": {"reason": reason, "missing": missing, "detail": list(detail)}}


OUTPUT_FILES = ("REPORT.md", "results.json", "cells.csv", "run_meta.json")


def clear_previous_outputs(out: str) -> list:
    """Remove the generated outputs of an earlier run (REPORT.md, results.json,
    cells.csv, run_meta.json) so a run that fails or crashes can never leave a
    stale report looking current. The raw-data cache and the calibration folder
    are never touched. Returns the names removed."""
    removed = []
    for name in OUTPUT_FILES:
        path = os.path.join(out, name)
        if os.path.exists(path):
            os.remove(path)
            removed.append(name)
    return removed


def run_study(cfg: StudyConfig, *, fetcher=None, sleep=None, log=None) -> int:
    """The whole study. Returns the process exit code (0 done, 3 failed closed)."""
    log = log or _default_log
    sleep = sleep or time.sleep
    fetcher = fetcher or real_fetcher
    t_start = _utcnow()
    cutoff = check_cutoff(cfg.cutoff)
    specs = resolve_universe(cfg.universe)
    out = cfg.out
    raw_dir = os.path.join(out, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    for name in clear_previous_outputs(out):
        log(f"removed the stale output {name} of an earlier run")
    manifest_path = os.path.join(raw_dir, "manifest.json")
    needs = required_needs(specs)
    log(f"study start: {len(specs)} trigger series, {len(needs)} downloads, cutoff "
        f"{cutoff.date()}, B={cfg.B}, B_map={cfg.B_map}, workers={cfg.workers}, "
        f"skip_fetch={cfg.skip_fetch}")

    # ---- (a) acquisition -------------------------------------------------
    manifest_files = load_manifest(manifest_path)
    failures = {}
    if cfg.skip_fetch:
        log("--skip-fetch: using the cached raw files (digests are verified)")
    else:
        files, failures = fetch_all(needs, fetcher, raw_dir, sleep=sleep, pause=cfg.fetch_pause,
                                    retry_wait=cfg.retry_wait, log=log)
        manifest_files.update(files)
        write_manifest(manifest_path, manifest_files)
    frames, n_post = {}, {}
    for need in needs:
        key = (need.source, need.symbol)
        if key in failures:
            continue
        try:
            df = parse_raw(need.source, load_cached(raw_dir, manifest_files, need))
        except (CacheError, ValueError) as exc:
            failures[key] = (f"cached file unusable: {exc}" if isinstance(exc, CacheError)
                             else f"cached file does not parse: {exc}")
            continue
        sealed, n = seal_frame(df, cutoff)
        if len(sealed) == 0:
            failures[key] = "no rows on or before the cutoff"
            continue
        frames[key], n_post[key] = sealed, n

    built_series, vehicle_series = {}, {}
    for spec in specs:                                    # unusable data fails closed, not with a crash
        key = (spec.source, spec.symbol)
        if key in failures or (spec.reference is not None and spec.reference in failures):
            continue
        try:
            built_series[spec.name] = build_trigger_series(spec, frames, cutoff, n_post)
        except ValueError as exc:
            failures[key] = f"unusable data: {exc}"
    for need in needs:
        key = (need.source, need.symbol)
        if need.kind in ("vehicle", "core") and key not in failures:
            try:
                vehicle_series[need.symbol] = build_vehicle_series(need.symbol, frames, cutoff,
                                                                   n_post)
            except ValueError as exc:
                failures[key] = f"unusable data: {exc}"

    missing = []
    for need in needs:
        key = (need.source, need.symbol)
        if key in failures:
            missing.append({"name": need.name, "kind": need.kind, "source": need.source,
                            "symbol": need.symbol, "reason": failures[key]})
    listed = {m["name"] for m in missing}
    for spec in specs:                                   # JPY=X refuses to exist without DEXJPUS
        if spec.reference is not None and spec.reference in failures and spec.name not in listed:
            missing.append({"name": spec.name, "kind": "trigger", "source": spec.source,
                            "symbol": spec.symbol,
                            "reason": f"requires the {spec.reference[1]} external reference "
                                      f"(the same-day reference check), which is missing: the study refuses to build "
                                      f"{spec.name} without it"})
    manifest_by_name = {n.name: dict(manifest_files[raw_filename(n.source, n.symbol)],
                                     file=raw_filename(n.source, n.symbol))
                        for n in needs if raw_filename(n.source, n.symbol) in manifest_files}
    if not any(s.role == "confirmatory" for s in specs) and not missing:
        results = _failed_results(cfg, specs, [], "the universe contains no confirmatory-family "
                                                   "series, so the confirmatory max-t test cannot run")
        _finish(cfg, specs, results, {}, manifest_by_name, t_start)
        log("FAILED CLOSED: no confirmatory-family series in the universe")
        return 3
    if missing:
        results = _failed_results(
            cfg, specs, missing, "required data could not be obtained",
            [f"{m['name']} ({m['source']}:{m['symbol']}): {m['reason']}" for m in missing])
        _finish(cfg, specs, results, {}, manifest_by_name, t_start)
        log("FAILED CLOSED: " + "; ".join(m["name"] for m in missing) + " missing")
        return 3

    # ---- (b) series ------------------------------------------------------
    series_list = [built_series[s.name] for s in specs]
    by_name = {s.name: s for s in series_list}
    spec_by = {s.name: s for s in specs}
    for s in series_list:
        log(f"series {s.name}: {s.n} sessions {s.dates[0].date()} to {s.dates[-1].date()}")
    stage_failures = []

    def get_vehicle(ticker):
        return vehicle_series[ticker]

    def fail(stage, item, exc):
        stage_failures.append({"stage": stage, "item": item,
                               "reason": f"{type(exc).__name__}: {exc}"})
        log(f"STAGE FAILED ({stage}, {item}): {type(exc).__name__}: {exc}")

    roll_gap = []
    for fut_name, etf in ROLL_GAP_PAIRS:
        if fut_name in by_name and ("yahoo", etf) in frames:
            roll_gap.append(roll_gap_diagnostic(by_name[fut_name], get_vehicle(etf)))
    vehicle_first_dates = {}
    for spec in specs:
        for side in sfs.SIDES:
            v = sfa.VEHICLE_TABLE.get((spec.name, side))
            if v is not None and v.ticker not in vehicle_first_dates:
                obs = frames[("yahoo", v.ticker)].index.min()
                vehicle_first_dates[v.ticker] = {
                    "table": v.first_date.strftime("%Y-%m-%d"),
                    "observed": obs.strftime("%Y-%m-%d"),
                    "matches": bool(obs == v.first_date)}

    # ---- (c) exploratory map --------------------------------------------
    log("map: 288 cells x 2 sides per series with the same-cell null (this is the long step)")
    mjs = run_jobs(map_job, [(s, cfg.B_map) for s in series_list], cfg.workers)
    mj_by = {m["name"]: m for m in mjs}
    cells_df = pd.concat([m["frame"] for m in mjs], ignore_index=True)
    scored = sfa.plateau_scores(cells_df)
    tops = sfa.top_cells(scored)
    ranked = sfa.rank_cells(scored)
    log(f"map done: {len(cells_df)} cells, {int(scored['plateau_score'].notna().sum())} with a "
        "plateau score")

    # null stability pass rate for every reported cell (same rule, N1 null cells)
    signals_cache = {}

    def signals_of(name):
        if name not in signals_cache:
            signals_cache[name] = sfs.compute_triggers(by_name[name])
        return signals_cache[name]

    reported = {}
    for frame in (tops["overall"], tops["per_pair"]):
        for row in frame.itertuples():
            reported[(row.series, row.trigger, row.side, row.entry, row.H, row.tp, row.sl)] = None
    for key in list(reported):
        nm, trig, side, entry, H, tp, slv = key
        cell = cell_of(entry, H, tp, slv)
        nc = sfa.null_cells_for_stability(by_name[nm], trig, side, cell, B=cfg.B_map,
                                          seed=sfn.MAP_SEED)
        reported[key] = sfa.null_stability_pass_rate(nc)

    def top_rows(frame):
        rows = []
        for r in frame.to_dict(orient="records"):
            d = _row_dict(r)
            k = (r["series"], r["trigger"], r["side"], r["entry"], r["H"], r["tp"], r["sl"])
            ns = reported.get(k) or {}
            d["null_stab_pass_rate"] = json_safe(ns.get("pass_rate"))
            d["null_stab_pass_rate_given_positive"] = json_safe(ns.get("pass_rate_given_positive"))
            rows.append(d)
        return rows

    top_overall = top_rows(tops["overall"])
    top_per_pair = {}
    for (nm, side), grp in tops["per_pair"].groupby(["series", "side"], sort=False):
        top_per_pair[f"{nm}|{side}"] = top_rows(grp)

    # ---- (d) confirmatory test ------------------------------------------
    conf_series = [by_name[s.name] for s in specs if s.role == "confirmatory"]
    log(f"confirmatory test: {len(conf_series)} series, B={cfg.B}")
    obs = sfn.observe_family(conf_series)
    conf = sfn.run_confirmatory_test(obs, B=cfg.B)
    for line in conf.report_lines():
        log("  " + line)
    tkey = conf.tstar_key
    conf_dict = conf.to_dict()
    conf_dict["report_lines"] = conf.report_lines()
    conf_dict["groups"] = json_safe(conf.to_frame().to_dict(orient="records"))
    conf_dict = json_safe(conf_dict)

    # ---- vehicle tests ---------------------------------------------------
    scope = sfa.vehicle_test_scope(scored, tstar_group=tkey)
    scope_out, veh_results = [], {}
    for e in scope:
        item = {"role": e["role"], "series": e["series"], "side": e["side"],
                "trigger": e["trigger"], "cell": cell_key(e["cell"]), "vehicle": e["vehicle"],
                "position": e["position"], "run": bool(e["run"]),
                "plateau_score": e["plateau_score"], "result": None, "reason": None}
        if not e["run"]:
            item["reason"] = "no tradable vehicle for this pair"
        else:
            ser = by_name[e["series"]]
            dates = pd.DatetimeIndex(mj_by[e["series"]]["episodes"][f"{e['trigger']}|{e['side']}"]["dates"])
            try:
                vres = sfa.vehicle_test_for_pair(e["series"], e["side"], get_vehicle(e["vehicle"]),
                                                 dates, e["cell"], B=cfg.B, seed=sfs.N1_SEED,
                                                 extra_lag=ser.extra_lag)
                veh_results[(e["series"], e["side"], e["trigger"])] = (e, vres)
                item["result"] = _vehicle_result_dict(vres)
                log(f"vehicle {e['series']} {e['side']} on {e['vehicle']}: n={vres.n}, excess "
                    f"{vres.excess_pp:+.2f} pp, null percentile {vres.null_percentile:.2f}")
            except sfs.SealedWindowError:
                raise
            except Exception as exc:
                fail("vehicle test", f"{e['series']} {e['side']} on {e['vehicle']}", exc)
                item["reason"] = f"vehicle test failed: {type(exc).__name__}: {exc}"
        scope_out.append(item)
    tstar_scope = next((i for i in scope_out if i["role"] in ("tstar", "tstar+plateau_top3")), None)
    tstar_vres = None
    if tkey is not None and tkey in veh_results:
        tstar_vres = veh_results[tkey][1]
    candidate = sfa.candidate_definition(conf.p_headline if tkey is not None else None, tstar_vres,
                                         series=tkey[0] if tkey else None)
    cand_dict = json_safe(dataclasses.asdict(candidate))

    # ---- ABI-1 transfer lines -------------------------------------------
    spy_frame = frames[("yahoo", CORE_SYMBOL)]
    abi_t = None
    if tkey is None:
        abi_t = {"status": "not_run", "reason": "no T* group (no tested group)"}
    elif tstar_vres is None:
        abi_t = {"status": "not_run", "reason": "the vehicle test of the T* pair was not run"}
    else:
        e, vres = veh_results[tkey]
        vser = get_vehicle(e["vehicle"])
        try:
            abi_t = abi1_line(series_name=e["series"], side=e["side"], trigger=e["trigger"],
                              cell=e["cell"], ticker=e["vehicle"], position=e["position"],
                              signal_dates=vres.anchor_dates, vehicle_series=vser,
                              vehicle_frame=frames[("yahoo", e["vehicle"])], spy_frame=spy_frame,
                              cutoff=cutoff)
        except sfs.SealedWindowError:
            raise
        except Exception as exc:
            fail("ABI-1 transfer line", "T* pair", exc)
            abi_t = {"status": "not_computed", "reason": f"{type(exc).__name__}: {exc}"}
    abi_p = {"status": "not_run", "reason": "no vehicle-eligible pair has a plateau score"}
    for r in ranked.itertuples():
        v = sfa.VEHICLE_TABLE.get((r.series, r.side))
        if v is None:
            continue
        cell = cell_of(r.entry, r.H, r.tp, r.sl)
        ser = by_name[r.series]
        vser = get_vehicle(v.ticker)
        dates = pd.DatetimeIndex(mj_by[r.series]["episodes"][f"{r.trigger}|{r.side}"]["dates"])
        try:
            sig = vehicle_signal_dates(vser, dates, ser.extra_lag)
            abi_p = abi1_line(series_name=r.series, side=r.side, trigger=r.trigger, cell=cell,
                              ticker=v.ticker, position=v.position, signal_dates=sig,
                              vehicle_series=vser, vehicle_frame=frames[("yahoo", v.ticker)],
                              spy_frame=spy_frame, cutoff=cutoff)
        except sfs.SealedWindowError:
            raise
        except Exception as exc:
            fail("ABI-1 transfer line", "top plateau cell", exc)
            abi_p = {"status": "not_computed", "reason": f"{type(exc).__name__}: {exc}"}
        break
    abi1 = {"tstar": abi_t, "top_plateau": abi_p}

    # ---- put screens -----------------------------------------------------
    iv_series = {n: pd.Series(by_name[n].p, index=by_name[n].dates)
                 for n in ("^VVIX", "^OVX", "^GVZ") if n in by_name}
    spy_close = pd.Series(spy_frame["Close"].to_numpy(dtype=float), index=spy_frame.index)
    rv_spy = pd.Series(sfa.realized_vol_series(spy_close.to_numpy()), index=spy_close.index)
    put_targets, seen_pairs = [], set()
    if tkey is not None:
        put_targets.append((tkey[0], tkey[2], tkey[1], sfs.PRIMARY_CELL, "tstar"))
        seen_pairs.add((tkey[0], tkey[1]))
    for r in ranked.itertuples():
        if r.side == "up_fade" and (r.series, r.side) not in seen_pairs:
            seen_pairs.add((r.series, r.side))
            put_targets.append((r.series, r.trigger, r.side, cell_of(r.entry, r.H, r.tp, r.sl),
                                "plateau_top"))
    put_out = []
    for nm, trig, side, cell, role in put_targets:
        if side != "up_fade":
            put_out.append({"series": nm, "side": side, "trigger": trig, "cell": cell_key(cell),
                            "role": role, "H": cell[1], "status": "not_applicable",
                            "reason": f"the T* side is {side}: a put is bought after an "
                                      "up-spike, so the put screen does not apply"})
            continue
        try:
            ser = by_name[nm]
            t0 = np.asarray(mj_by[nm]["episodes"][f"{trig}|{side}"]["t0"], dtype=np.int64)
            mn = sfn.n1_cell_null(ser, trig, side, cells=[cell], B=cfg.B_map, seed=sfn.MAP_SEED,
                                  signals=signals_of(nm), collect_draws=True)
            base_t0 = mn.anchors.ravel() if (mn.anchors is not None and mn.anchors.size) else None
            proxy = sfa.STUDY_SERIES_IV_PROXY.get(nm)
            res = sfa.put_screen_for_cell(ser, t0, cell, side="up_fade", baseline_t0=base_t0,
                                          iv_close=iv_series.get(proxy) if proxy else None)
            d = _put_dict(nm, side, trig, cell, role, res)
            d["market_k_proxy"] = proxy if proxy in iv_series else None
            d["market_k_summary"] = res["market_k_summary"]
            vs = None
            if "^VIX" in by_name:
                ed = res["inputs"]["entry_dates"]
                rv_at = rv_spy.reindex(pd.DatetimeIndex(ed)).to_numpy(dtype=float)
                vs = sfa.market_k_summary(sfa.market_k_at_entries(
                    pd.Series(by_name["^VIX"].p, index=by_name["^VIX"].dates), ed, rv_at))
            d["market_k_vix_over_spy"] = vs
            put_out.append(d)
        except sfs.SealedWindowError:
            raise
        except Exception as exc:
            fail("put screen", f"{nm} {side}", exc)
            put_out.append({"series": nm, "side": side, "trigger": trig,
                            "cell": cell_key(cell), "role": role, "H": cell[1],
                            "status": "not_computed",
                            "reason": f"{type(exc).__name__}: {exc}"})

    # ---- assemble --------------------------------------------------------
    series_out = {s.name: _series_info(spec_by[s.name], s, mj_by[s.name], cells_df)
                  for s in series_list}
    vehicle_series_out = {}
    for n in needs:                    # data-quality rule: the largest moves of EVERY series are listed
        if n.kind in ("vehicle", "core"):
            vs = vehicle_series[n.symbol]
            vehicle_series_out[n.symbol] = {
                "role": n.kind, "n_rows": int(vs.n),
                "first_date": vs.dates[0].strftime("%Y-%m-%d"),
                "last_date": vs.dates[-1].strftime("%Y-%m-%d"),
                "counts": vs.counts(),
                "largest_returns": [{"date": r.date.strftime("%Y-%m-%d"),
                                     "prev_date": r.prev_date.strftime("%Y-%m-%d"),
                                     "log_return": float(r.log_return),
                                     "abs_log_return": float(r.abs_log_return)}
                                    for r in sfs.largest_abs_returns(vs).itertuples()]}
    files_out = {}
    for n in needs:
        fn = raw_filename(n.source, n.symbol)
        fr = frames[(n.source, n.symbol)]
        files_out[n.name] = {"kind": n.kind, "source": n.source, "symbol": n.symbol,
                             "file": fn, "sha256": manifest_files[fn]["sha256"],
                             "rows_within_cutoff": int(len(fr)),
                             "first_date": fr.index.min().strftime("%Y-%m-%d"),
                             "last_date": fr.index.max().strftime("%Y-%m-%d")}
    tot = Counter()
    for nm, s in series_out.items():
        for k in ("rows", "missing_rows_dropped", "pre_start_dropped", "post_cutoff_dropped",
                  "nonpositive", "external_corrections"):
            tot[k] += s["counts"][k]
        for g in s["groups"]:
            tot["flagged_days"] += g["n_flagged_days"]
            tot["episodes"] += g["n_episodes"]
            tot["kept"] += g["n_kept"]
            tot["dropped_censored"] += g["dropped_censored"]
        for k, v in s["cell_status_totals"].items():
            tot[k] += v

    family_rows = []
    for spec in specs:
        if spec.role != "confirmatory":
            continue
        for side in sfs.SIDES:
            tested = [(obs.groups[k].t, obs.groups[k].n, k) for k in obs.tested
                      if k[0] == spec.name and k[1] == side]
            best = sfn.select_tstar(tested)
            best_cell = ranked[(ranked["series"] == spec.name) & (ranked["side"] == side)]
            bc = best_cell.iloc[0] if len(best_cell) else None
            veh = next((v for v in scope_out if v["series"] == spec.name and v["side"] == side
                        and v["run"] and v["result"] is not None), None)
            row = {"series": spec.name, "side": side,
                   "trigger": best[1][2] if best else None,
                   "n": best[2] if best else None,
                   "mean_x": obs.groups[best[1]].mean_x if best else None,
                   "t": best[0] if best else None,
                   "is_tstar": bool(best and tkey is not None and best[1] == tkey),
                   "adj_p_headline": (conf.per_pair.get((spec.name, side), {}).get("p_headline")
                                      if best else None),
                   "vehicle": (sfa.VEHICLE_TABLE[(spec.name, side)].ticker),
                   "vehicle_status": "run" if veh else "not_run",
                   "vehicle_excess_pp": veh["result"]["excess_pp"] if veh else None,
                   "vehicle_null_percentile": veh["result"]["null_percentile"] if veh else None,
                   "vehicle_n": veh["result"]["n"] if veh else None,
                   "vehicle_cell": veh["cell"] if veh else None,
                   "best_plateau_cell": (f"{bc['trigger']} | {bc['entry']} | H {bc['H']} | "
                                         f"TP {bc['tp']} | SL {bc['sl']}") if bc is not None else None,
                   "best_plateau_score": float(bc["plateau_score"]) if bc is not None else None}
            family_rows.append(row)

    tstar_series = tkey[0] if tkey else None
    recommendation = build_recommendation(cand_dict, abi_t, tstar_series=tstar_series)
    calibration = _read_calibration(out)
    vres_summary = None
    if tstar_scope and tstar_scope["result"]:
        r = tstar_scope["result"]
        vres_summary = {"ticker": tstar_scope["vehicle"], "position": tstar_scope["position"],
                        "n": r["n"], "excess_pp": r["excess_pp"], "excess_sigma": r["excess_sigma"],
                        "null_percentile": r["null_percentile"], "hit_rate": r["hit_rate"],
                        "mean_pnl_pp": r["mean_pnl_pp"],
                        "unconditional_mean_pnl_pp": r["unconditional_mean_pnl_pp"],
                        "underpowered": r["underpowered_on_vehicle"], "reps": r["null_reps_used"]}
    headline = {
        "p_headline": conf_dict["p_headline"], "p_n1": conf_dict["p_n1"],
        "n2_sensitivity": {k: (conf_dict.get("n2_sensitivity") or {}).get(k)
                           for k in ("label", "gating", "p_n2", "mc_se_n2", "available_for_tstar",
                                     "offsets", "skipped", "p_n2_covered")},
        "decision": conf_dict["decision"], "mc_se_headline": conf_dict["mc_se_headline"],
        "B": int(cfg.B), "tstar": conf_dict["tstar"], "tstar_key": conf_dict["tstar_key"],
        "tstar_n": conf_dict["tstar_n"],
        "tstar_series": tkey[0] if tkey else None, "tstar_side": tkey[1] if tkey else None,
        "tstar_trigger": tkey[2] if tkey else None, "vehicle": vres_summary,
        "candidate": cand_dict, "top5_plateau": top_overall[:TOP_HEADLINE_CELLS],
        "abi1_tstar": _abi_summary(abi_t), "abi1_top_plateau": _abi_summary(abi_p),
        "calibration_status": "complete" if calibration else "pending"}
    results = {
        "schema": SCHEMA, "status": "completed",
        "spec": {"file": SPEC_FILE, "version": SPEC_VERSION},
        "cutoff": cutoff.strftime("%Y-%m-%d"), "params": _params(cfg, specs),
        "data": {"files": files_out, "labels": DATA_LABELS},
        "series": series_out, "vehicle_series": vehicle_series_out, "roll_gap": roll_gap, "vehicle_first_dates": vehicle_first_dates,
        "confirmatory": conf_dict, "family_table": family_rows,
        "map": {"n_series": len(series_list), "n_groups": len(series_list) * 8,
                "n_cells": int(len(cells_df)), "n_eligible_cells": int(cells_df["eligible"].sum()),
                "n_with_plateau": int(scored["plateau_score"].notna().sum()),
                "top_overall": top_overall, "top_per_pair": top_per_pair,
                "null_column_note": "context only"},
        "vehicle": {"scope": scope_out, "candidate": cand_dict},
        "put_screen": put_out, "abi1": abi1,
        "counts": {"totals": dict(tot)}, "headline": headline, "stage_failures": stage_failures,
        "recommendation": recommendation,
        "open_choices": list(sfn.OPEN_CHOICES) + list(sfa.OPEN_CHOICES) + list(DRIVER_OPEN_CHOICES),
        "failure": None}

    cells_out = scored.copy()
    cells_out["worst_date"] = cells_out["worst_date"].astype(str)
    cells_out["first_usable_date"] = cells_out["first_usable_date"].astype(str)
    cells_out.to_csv(os.path.join(out, "cells.csv"), index=False, lineterminator="\n")
    _finish(cfg, specs, results, {"elapsed_note": "see started_utc and finished_utc"},
            manifest_by_name, t_start)
    log(f"study done: headline p = {conf_dict['p_headline']}, T* = {conf_dict['tstar_key']}")
    if stage_failures:
        log(f"INCOMPLETE: {len(stage_failures)} stage failure(s) recorded in results.json")
        return 4
    return 0


DRIVER_OPEN_CHOICES = (
    "Driver: the vehicle test's trigger anchors are the trigger group's kept episodes "
    "(gap20, uniformly censored on the trigger series' own calendar); the vehicle's own "
    "status then decides which of them trade.",
    "Driver: the put screen runs for the T* pair (only if its side is up_fade: a put is "
    "bought after an up-spike) and for the best-ranked cell of every up-side pair; its "
    "random-date baseline pools the same-cell N1 draws (B_map replications, the map seed).",
    "Driver: the ABI-1 descriptive line is built for the best-ranked cell among "
    "vehicle-eligible pairs (its signal bars come from the same anchor mapping the "
    "vehicle test uses); a calendar mismatch or a non-positive price is reported as "
    "not computed with the reason, never patched around.",
    "Driver: the roll-gap ratio uses the standard deviation of the FUTURES series' daily "
    "log returns over the overlap as 'its own daily standard deviation'; returns are "
    "taken between consecutive common valid sessions.",
    "Driver: the headline p and the confirmatory dictionary are taken verbatim from the null "
    "module; the report shows the N2 translation null as a separate sensitivity.",
    "Driver: an exploratory-only series that cannot be obtained also fails the study closed "
    "(the study fixes the whole universe in advance, so a partial one is a silent change).",
    "Driver: the recommendation's condition (b) requires the excess Sharpe to reach the LARGER "
    "of the nominal in-sample bar (about 1.02) and the exact bar for the account's own length "
    "(the pipeline's deflated Sharpe at Normal moments); both are printed.",
    "Driver: the roll-off share is the fraction of threshold-crossing days on which the close "
    "was not a fresh 20-session extreme (the core module's own definition, listed there as an "
    "open choice).",
    "Driver: the vehicle test uses B = the confirmatory B (500) and the frozen N1 seed with a "
    "stream keyed by the vehicle name; the cell cluster bootstrap uses B = 2000 and its own "
    "frozen seed (both in the core module).",
    "Driver: the AR(1) half-life is an OLS fit with intercept on the log level over valid "
    "consecutive pairs; the roll-gap check needs at least 30 overlapping returns.",
    "Driver: a later stage (vehicle test, put screen, ABI-1 line) that raises is recorded in "
    "results.json as a stage failure, the report says the run is INCOMPLETE and the exit code "
    "is 4; the headline is never discarded, but a stage failure is a defect to fix and re-run.",
    "Driver: yfinance history is requested with period max and auto_adjust True; vendor "
    "dividend adjustment after the cutoff rescales the whole pre-cutoff history by a "
    "constant, which cancels in every return, ratio and comparison used, and the sealed "
    "loader truncates before any computation.",
)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def render_only(out: str) -> int:
    """Rebuild REPORT.md from a finished results.json (plus run_meta.json, the
    manifest and the calibration file) WITHOUT recomputing anything: results.json
    and cells.csv are not touched. Exit 2 if there is no results.json."""
    path = os.path.join(out, "results.json")
    if not os.path.exists(path):
        print(f"REFUSED: {path} does not exist; there is nothing to render", flush=True)
        return 2
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    meta_path = os.path.join(out, "run_meta.json")
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    files = load_manifest(os.path.join(out, "data", "raw", "manifest.json"))
    by_name = {}
    for name, ent in ((results.get("data") or {}).get("files") or {}).items():
        fn = ent.get("file")
        if fn in files:
            by_name[name] = dict(files[fn], file=fn)
    text = rep.render_report(results, meta=meta, calibration=_read_calibration(out),
                             manifest=by_name, robustness=_read_robustness(out))
    with open(os.path.join(out, "REPORT.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"REPORT.md re-rendered from {path}", flush=True)
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=DEFAULT_OUT, help="output folder (default runs/spike-fade-study)")
    ap.add_argument("--B", type=int, default=sfn.NULL_B, help="null replications (500 replications)")
    ap.add_argument("--B-map", dest="B_map", type=int, default=sfn.MAP_B,
                    help="replications of the map's same-cell null (200 replications)")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS, help="worker processes (at most 4)")
    ap.add_argument("--skip-fetch", action="store_true", help="use the cached raw data")
    ap.add_argument("--universe", default=None, help="comma list of trigger series (tests)")
    ap.add_argument("--cutoff", default=sfs.DISCOVERY_CUTOFF.strftime("%Y-%m-%d"),
                    help="last usable date; a date after 2023-06-30 is refused")
    ap.add_argument("--force", action="store_true",
                    help="allow a new run to replace an existing COMPLETED results.json")
    ap.add_argument("--render-only", action="store_true",
                    help="rebuild REPORT.md from an existing results.json; recompute nothing")
    ap.add_argument("--fetch-pause", type=float, default=FETCH_PAUSE)
    ap.add_argument("--retry-wait", type=float, default=RETRY_WAIT)
    return ap.parse_args(argv)


def main(argv=None, *, fetcher=None, sleep=None) -> int:
    args = parse_args(argv)
    if args.render_only:
        return render_only(os.path.abspath(args.out))
    try:
        cutoff = check_cutoff(pd.Timestamp(args.cutoff))
    except CutoffRefused as exc:
        print(f"REFUSED: {exc}", flush=True)
        return 2
    try:
        resolve_universe(args.universe)
    except ValueError as exc:
        print(f"REFUSED: {exc}", flush=True)
        return 2
    prior = os.path.join(os.path.abspath(args.out), "results.json")
    if not args.force and os.path.exists(prior):
        try:
            with open(prior, encoding="utf-8") as f:
                done = json.load(f).get("status") == "completed"
        except (OSError, ValueError):
            done = False
        if done:
            print(f"REFUSED: {prior} holds a completed run; pass --force to replace it "
                  "(a re-run after a defect must be deliberate)", flush=True)
            return 2
    cfg = StudyConfig(out=os.path.abspath(args.out), B=args.B, B_map=args.B_map,
                      workers=max(1, min(args.workers, MAX_WORKERS)), skip_fetch=args.skip_fetch,
                      universe=args.universe, cutoff=cutoff, fetch_pause=args.fetch_pause,
                      retry_wait=args.retry_wait, argv=tuple(argv or sys.argv[1:]),
                      force=args.force)
    try:
        return run_study(cfg, fetcher=fetcher, sleep=sleep)
    except Exception:                                    # an unexpected error is a crash, loudly
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
