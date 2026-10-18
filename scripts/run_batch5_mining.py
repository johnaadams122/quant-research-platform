"""Batch5 directional mining driver.

Runs the curated batch5 candidates (an ``accepted.json`` curation file,
passed via ``--accepted``) through the SAME pre-registered scoring machinery
as scripts/run_batch3_mining.py -- ``compute_shared_correction`` +
``score_batch`` + ``execute_batch_run`` are imported, never copied, so this
script owns no correction arithmetic and no ledger write of its own.

What differs here from batch3:

* Candidates are self-describing modules under ``src/mining/batch5/``
  (one per accepted family, auto-discovered; the required contract
  fields are listed in ``_CONTRACT_FIELDS`` below). Every contract
  field is validated and ``PARAMETERS`` must equal the catalog entry
  byte-for-byte (JSON, sorted keys) -- any failure SKIPS that candidate
  with a recorded reason; nothing is ever silently dropped.
* Six shapes: ``single_long`` / ``single_inverse_product`` /
  ``panel_breadth`` (one traded instrument, run_single_asset_windows),
  ``pooled_single_asset`` / ``event_pooled`` (per-ticker
  run_single_asset_windows, then trade-level pooling -- ONE trial per
  family, never one per ticker) and ``rotation_two_asset``
  (run_rotation_windows, exactly as batch3's compute_candidate).
* Costs come ONLY from the dated Friday-close quotes file: a missing,
  suspect, mis-ordered or >300 bps quote skips the candidate as
  ``missing_data``. No cost is ever invented.
* A REPORT-ONLY directional-support panel per candidate
  (mining.benchmark_relative): alpha/beta vs own buy-and-hold and vs SPY,
  time-matched SPY excess per trade (in-sample and holdout), a full-window
  summary, and the DSR formula applied to the DAILY mark-to-market series.
  It lives under ``item["directional_panel"]`` and never influences stage,
  dsr_value or any ledger record.
* ``--retest`` lists already-pinned family_ids that may be re-scored against
  their pinned window with no new trial charge (score_batch's
  ``retest_family_ids`` hook).

Pooled-shape conventions used by the directional panel, stated once:
strategy daily return = equal-weight MEAN of the per-ticker mark-to-market
sleeves aligned on the union calendar with 0.0 fill; own buy-and-hold =
equal-weight mean of the pool's buy-and-hold series on the same basis;
time-matched SPY excess = over the POOLED trades. Rotation: the two legs
never overlap, so their per-leg sleeves are SUMMED; a switch-day exit is
marked through that day's close although the engine exits at the open --
a documented report-only approximation, not a change to any score.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
import pkgutil
import re
import sys
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
SCRIPTS_ROOT = os.path.join(REPO_ROOT, "scripts")
for _path in (SRC_ROOT, SCRIPTS_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import data_pipeline
import run_batch3_mining as batch3
from mining import (backtest_engine, benchmark_relative, catalog, constants,
                    cost_tax_inputs, panel, pooling, reporting, trial_ledger)


SHAPES = ("single_long", "single_inverse_product", "rotation_two_asset",
          "pooled_single_asset", "panel_breadth", "event_pooled")
SINGLE_SHAPES = frozenset({"single_long", "single_inverse_product",
                           "panel_breadth"})
POOLED_SHAPES = frozenset({"pooled_single_asset", "event_pooled"})
ROTATION_SHAPE = "rotation_two_asset"
QUOTE_GATE_BPS = 300.0
MARKET_BENCHMARK = "SPY"
DIRECTIONAL_SUPPORT_T_ALPHA = 2.0

DEFAULT_RUN_ID = "batch5"
DEFAULT_TESTED_DATE = datetime.date.today().isoformat()
LEDGER_PATH = batch3.LEDGER_PATH
CATALOG_PATH = batch3.CATALOG_PATH
RESEARCH_ROOT = batch3.RESEARCH_ROOT
RESULTS_LOG_PATH = batch3.RESULTS_LOG_PATH
NEAR_MISS_LOG_PATH = batch3.NEAR_MISS_LOG_PATH
DEFAULT_MODULES_DIR = os.path.join(SRC_ROOT, "mining", "batch5")
BATCH5_DIR = os.path.join(RESEARCH_ROOT, DEFAULT_RUN_ID)
DEFAULT_ACCEPTED = os.path.join(BATCH5_DIR, "curation", "accepted.json")
DEFAULT_QUOTES = os.path.join(BATCH5_DIR, "curation", "quotes.json")

_CONTRACT_FIELDS = ("FAMILY_ID", "SHAPE", "TRADED", "SIGNAL", "SNAPSHOTS",
                    "BENCHMARK", "PARAMETERS", "EXPECTED_MIN_TRADES")

EXTERNAL_PATH_PREFIX = "<external>"


def portable_path(path: str, *, repo_root: str | None = None) -> str:
    """The form of ``path`` that is safe to write into a committed results
    file: repo-relative with forward slashes, so the file never names the
    workstation (drive letter, user profile) it was produced on.

    A path outside the repo -- or on another drive -- has no repo-relative
    form, so only its file name is kept, behind ``<external>/``. A relative
    ``path`` is resolved against the working directory first. ``repo_root``
    defaults to this module's ``REPO_ROOT``, read at call time.
    """
    root = os.path.abspath(repo_root or REPO_ROOT)
    absolute = os.path.abspath(path)
    try:
        relative = os.path.relpath(absolute, root)
    except ValueError:  # a different drive: no relative form exists
        relative = None
    if (relative is None or relative == os.pardir
            or relative.startswith(os.pardir + os.sep)):
        return f"{EXTERNAL_PATH_PREFIX}/{os.path.basename(absolute)}"
    return relative.replace(os.sep, "/")


# --------------------------------------------------------------- inputs

def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_accepted(path: str) -> dict:
    payload = _read_json(path)
    accepted = payload.get("accepted")
    if not isinstance(accepted, list):
        raise ValueError(f"{path}: 'accepted' must be a list")
    seen = set()
    for index, candidate in enumerate(accepted):
        family_id = candidate.get("family_id") if isinstance(candidate, dict) else None
        if not isinstance(family_id, str) or not family_id:
            raise ValueError(f"{path}: accepted[{index}] lacks a family_id")
        if family_id in seen:
            raise ValueError(f"{path}: duplicate accepted family_id {family_id}")
        seen.add(family_id)
    return payload


def load_quotes(path: str) -> dict:
    payload = _read_json(path)
    if not isinstance(payload.get("quotes"), dict):
        raise ValueError(f"{path}: 'quotes' must be an object keyed by symbol")
    for field in ("sampled_at_local", "market_state"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise ValueError(f"{path}: '{field}' must be a non-empty string")
    return payload


def quotes_cost_source(quotes_payload: Mapping) -> str:
    return (f"{quotes_payload.get('source') or 'quotes file'} sampled "
            f"{quotes_payload['sampled_at_local']} "
            f"[{quotes_payload['market_state']}]")


SYNTHETIC_QUOTE_MARKER = "SYNTHETIC"


def refuse_synthetic_quotes(quotes_payload: Mapping, *, dry_run: bool) -> None:
    """A REAL run must price its pre-registered verdict off a MEASURED book.

    The dry run is exercised on a synthetic quote file so every candidate
    clears the cost gate and the whole pipeline runs end to end; scoring a
    real trial against invented spreads would record a verdict against a cost
    nobody observed. The dry-run-only file announces itself, and this refuses
    it outside a dry run."""
    if dry_run:
        return
    # Scan EVERY top-level field, not just a few: a marker in `notes`, in
    # any other top-level key, or on an individual quote must not pass
    # unrefused. A file that
    # announces itself anywhere is refused everywhere.
    for field, value in sorted(quotes_payload.items()):
        for text in _strings_in(value):
            if SYNTHETIC_QUOTE_MARKER in text.upper():
                raise ValueError(
                    f"refusing a real run against a {SYNTHETIC_QUOTE_MARKER} "
                    f"quotes file ({field} contains {text!r}): a recorded "
                    "trial must be priced from a measured quote sample. "
                    "Re-run with --dry-run, or pass a real intraday quote "
                    "file.")


def _strings_in(value, _depth: int = 0):
    """Every string reachable in a nested payload, depth-bounded."""
    if _depth > 4:
        return
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, inner in value.items():
            if isinstance(key, str):
                yield key
            yield from _strings_in(inner, _depth + 1)
    elif isinstance(value, (list, tuple)):
        for inner in value:
            yield from _strings_in(inner, _depth + 1)


# ------------------------------------------------------------ discovery

def _import_module_file(name: str, path: str):
    digest = hashlib.sha256(os.path.abspath(path).encode("utf-8")).hexdigest()[:12]
    unique = f"batch5_candidate_{digest}_{name}"
    spec = importlib.util.spec_from_file_location(unique, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(unique, None)
        raise
    return module


def _str_list(module, field: str, problems: list, *, nonempty: bool):
    value = getattr(module, field, None)
    if not isinstance(value, (list, tuple)) or any(
            not isinstance(item, str) or not item for item in value):
        problems.append(f"{field} must be a list of non-empty str")
        return None
    if nonempty and not value:
        problems.append(f"{field} must not be empty")
        return None
    if len(set(value)) != len(value):
        problems.append(f"{field} must not repeat a symbol")
        return None
    return list(value)


def validate_module_contract(module, *, repo_root: str = REPO_ROOT) -> dict:
    """Validate every contract field; raise ValueError naming EVERY problem
    (never just the first) so an implementer fixes them in one pass."""
    problems = []
    for field in _CONTRACT_FIELDS:
        if not hasattr(module, field):
            problems.append(f"missing {field}")
    family_id = getattr(module, "FAMILY_ID", None)
    if not isinstance(family_id, str) or not family_id:
        problems.append("FAMILY_ID must be a non-empty str")
    shape = getattr(module, "SHAPE", None)
    if shape not in SHAPES:
        problems.append(f"SHAPE must be one of {list(SHAPES)}, got {shape!r}")
    traded = _str_list(module, "TRADED", problems, nonempty=True)
    signal = _str_list(module, "SIGNAL", problems, nonempty=False)
    if traded is not None and signal is not None and set(traded) & set(signal):
        problems.append("TRADED and SIGNAL must not overlap")
    benchmark = getattr(module, "BENCHMARK", None)
    if not isinstance(benchmark, str) or not benchmark:
        problems.append("BENCHMARK must be a non-empty str")
    parameters = getattr(module, "PARAMETERS", None)
    if not isinstance(parameters, dict):
        problems.append("PARAMETERS must be a dict")
    else:
        try:
            json.dumps(parameters, sort_keys=True)
        except (TypeError, ValueError) as exc:
            problems.append(f"PARAMETERS must be JSON-serializable: {exc}")
    expected = getattr(module, "EXPECTED_MIN_TRADES", None)
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
        problems.append("EXPECTED_MIN_TRADES must be a non-negative int")
    snapshots = getattr(module, "SNAPSHOTS", None)
    if not isinstance(snapshots, dict):
        problems.append("SNAPSHOTS must be a dict (empty unless panel_breadth)")
        snapshots = None
    if shape == ROTATION_SHAPE:
        if traded is not None and len(traded) != 2:
            problems.append("rotation_two_asset requires exactly 2 TRADED "
                            "symbols [risk, defensive]")
        if not callable(getattr(module, "make_target_fn", None)):
            problems.append("rotation_two_asset requires a callable make_target_fn")
    elif shape in SHAPES:
        if not callable(getattr(module, "make_signal_fn", None)):
            problems.append(f"{shape} requires a callable make_signal_fn")
        if shape in SINGLE_SHAPES and traded is not None and len(traded) != 1:
            problems.append(f"{shape} requires exactly 1 TRADED symbol")
    snapshot_summary = {}
    if shape == "panel_breadth":
        if not snapshots:
            problems.append("panel_breadth requires a non-empty SNAPSHOTS dict")
        else:
            for name, spec in snapshots.items():
                if not isinstance(spec, dict) or not isinstance(
                        spec.get("path"), str) or not isinstance(
                        spec.get("sha256"), str):
                    problems.append(
                        f"SNAPSHOTS[{name!r}] must be a dict with str "
                        "'path' and 'sha256'")
                    continue
                path = spec["path"]
                if not os.path.isabs(path):
                    path = os.path.join(repo_root, path)
                if not os.path.exists(path):
                    problems.append(
                        f"SNAPSHOTS[{name!r}] path does not exist: "
                        f"{portable_path(path, repo_root=repo_root)}")
                    continue
                try:
                    loaded = panel.load_breadth_snapshot(
                        path, expect_sha256=spec["sha256"])
                except Exception as exc:  # any refusal is a contract failure
                    # panel.load_breadth_snapshot's own messages can embed
                    # this absolute path -- scrub it before it reaches a
                    # committed reason.
                    problems.append(
                        f"SNAPSHOTS[{name!r}] failed to load: "
                        f"{type(exc).__name__}: "
                        f"{str(exc).replace(path, portable_path(path, repo_root=repo_root))}")
                    continue
                snapshot_summary[name] = {
                    "path": portable_path(path, repo_root=repo_root),
                    "sha256": spec["sha256"], **loaded.describe()}
    elif snapshots:
        problems.append("SNAPSHOTS must be empty unless SHAPE is panel_breadth")
    if problems:
        raise ValueError("; ".join(problems))
    return {
        "family_id": family_id, "shape": shape, "traded": traded,
        "signal": signal, "benchmark": benchmark,
        "parameters": parameters, "expected_min_trades": expected,
        "snapshots": snapshot_summary,
        "module_file": getattr(module, "__file__", None),
    }


def discover_modules(modules_dir: str, *,
                     repo_root: str = REPO_ROOT) -> tuple[dict, list]:
    """Import every ``<modules_dir>/*.py`` (pkgutil discovery), validate
    its contract, and return ``({family_id: {"module", "spec"}}, problems)``
    -- an invalid module is recorded in ``problems`` with its reason and
    is NOT in the mapping."""
    modules_dir = os.path.abspath(modules_dir)
    if not os.path.isdir(modules_dir):
        raise FileNotFoundError(f"modules dir does not exist: {modules_dir}")
    found, problems = {}, []
    for info in sorted(pkgutil.iter_modules([modules_dir]), key=lambda m: m.name):
        if info.name.startswith("_") or info.ispkg:
            continue
        path = os.path.join(modules_dir, info.name + ".py")
        try:
            module = _import_module_file(info.name, path)
            spec = validate_module_contract(module, repo_root=repo_root)
        except Exception as exc:
            # An import-spec failure names this path directly, and a
            # candidate module's own top-level exception can too (via its
            # __file__) -- scrub it before it reaches a committed reason.
            reason = str(exc).replace(path, portable_path(path, repo_root=repo_root))
            problems.append({"module": info.name,
                             "path": portable_path(path, repo_root=repo_root),
                             "reason": f"{type(exc).__name__}: {reason}"})
            continue
        family_id = spec["family_id"]
        if family_id in found:
            problems.append({"module": info.name,
                             "path": portable_path(path, repo_root=repo_root),
                             "reason": f"duplicate FAMILY_ID {family_id} (already "
                                       f"provided by another module)"})
            continue
        found[family_id] = {"module": module, "spec": spec}
    return found, problems


def _canonical(parameters) -> str:
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def parameter_drift(module_parameters: dict, catalog_parameters) -> str | None:
    """None when identical as canonical JSON; otherwise the reason."""
    ours, theirs = _canonical(module_parameters), _canonical(catalog_parameters)
    if ours == theirs:
        return None
    return f"module PARAMETERS {ours} != catalog parameters {theirs}"


# ---------------------------------------------------------------- costs

def execution_costs_for(traded: list, quotes_payload: Mapping, *,
                        quote_gate_bps: float = QUOTE_GATE_BPS) -> dict:
    """One empirical round-trip cost per TRADED symbol from the dated
    quotes file, or ValueError naming the first symbol that fails the
    gate. Signal-only symbols never come through here."""
    quotes = quotes_payload["quotes"]
    source = quotes_cost_source(quotes_payload)
    selected = {}
    for symbol in traded:
        quote = quotes.get(symbol)
        if not isinstance(quote, dict):
            raise ValueError(f"{symbol}: no quote in the quotes file")
        if quote.get("suspect") is True:
            raise ValueError(f"{symbol}: quote flagged suspect in the quotes file")
        try:
            bid, ask = float(quote["bid"]), float(quote["ask"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{symbol}: quote missing numeric bid/ask") from exc
        if not (math.isfinite(bid) and math.isfinite(ask) and 0.0 < bid < ask):
            raise ValueError(
                f"{symbol}: quote must satisfy 0 < bid < ask (bid={bid}, ask={ask})")
        spread = cost_tax_inputs.compute_spread_bps(bid, ask)
        if spread > quote_gate_bps:
            raise ValueError(
                f"{symbol}: spread {spread:.4f} bps exceeds the {quote_gate_bps:.0f} "
                f"bps gate")
        cost = cost_tax_inputs.cost_bps_from_quotes(bid, ask, bid, ask)
        selected[symbol] = {
            "bid": bid, "ask": ask, "spread_bps": spread, "cost_bps": cost,
            "security_status": quote.get("security_status"),
            "source": source,
        }
    return selected


# ----------------------------------------------------------------- OHLC

def default_ohlc_loader(symbol: str):
    return data_pipeline.fetch_ohlc(symbol, start="max")


def _cache_name(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", symbol) + ".csv"


OHLC_COLUMNS = ("Open", "High", "Low", "Close")


def _drop_all_nan_rows(raw: pd.DataFrame) -> tuple:
    """Drop rows whose every OHLC field is blank, returning
    ``(frame, [dropped dates])``.

    A row with NO prices at all is a session the vendor never printed (the
    vendor's HUBB history carries one such all-blank row, dated 1977), which
    ``batch3._normalize_ohlc`` refuses outright -- costing a pooled family a
    member over a missing print. A row with SOME prices is a different thing:
    a bad print, which must keep failing closed, so it is left alone here."""
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return raw, []
    present = [column for column in OHLC_COLUMNS if column in raw.columns]
    if not present:
        return raw, []
    blank = raw[present].isna().all(axis=1)
    if not blank.any():
        return raw, []
    if "Date" in raw.columns:
        stamps = pd.to_datetime(raw.loc[blank, "Date"])
    else:
        stamps = pd.to_datetime(pd.Index(raw.index[blank]))
    dropped = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in stamps]
    return raw.loc[~blank], dropped


def load_frame(symbol: str, *, ohlc_loader: Callable, cache_dir: str,
               report: dict | None = None,
               require_positive: bool = True) -> tuple:
    """Normalized OHLC for ``symbol``; reuses ``<cache_dir>/<SYM>.csv`` when
    present, else fetches through ``ohlc_loader`` and writes the cache.
    Returns ``(frame, "cache" | "fetched")``. Any all-blank row dropped on the
    way is recorded in ``report['dropped_all_nan_rows']``.

    ``require_positive`` (default True) passes straight through to ``batch3._normalize_ohlc`` -- set False only
    for a symbol the caller already knows is never traded (a signal-only
    external series, e.g. a yield index that can legitimately go negative)."""
    path = os.path.join(cache_dir, _cache_name(symbol))
    sidecar = path + ".dropped.json"
    if os.path.exists(path):
        # A cache hit sees an already-cleaned frame, so the drop cannot be
        # re-derived from it. Without the sidecar the run report would state
        # that nothing was dropped -- losing the exact fact the field exists
        # to carry, and losing it permanently once the cache is warm.
        if report is not None and os.path.exists(sidecar):
            try:
                report["dropped_all_nan_rows"] = _read_json(sidecar)["dates"]
            except Exception as exc:
                report["dropped_rows_provenance_error"] = (
                    f"{type(exc).__name__}: {exc}")
        return (batch3._normalize_ohlc(pd.read_csv(path), symbol,
                                       require_positive=require_positive),
                "cache")
    raw = ohlc_loader(symbol)
    if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
        raise RuntimeError(f"{symbol}: OHLC loader returned no data")
    raw, dropped = _drop_all_nan_rows(raw)
    if dropped and report is not None:
        report["dropped_all_nan_rows"] = dropped
    frame = batch3._normalize_ohlc(raw, symbol, require_positive=require_positive)
    os.makedirs(cache_dir, exist_ok=True)
    temporary = path + ".tmp"
    frame.to_csv(temporary, index_label="Date")
    os.replace(temporary, path)
    if dropped:
        batch3._atomic_write_text(sidecar, json.dumps(
            {"symbol": symbol, "reason": "all OHLC fields blank", "dates": dropped},
            indent=1))
    return frame, "fetched"


def candidate_window(ledger_path: str, family_id: str, tested_date: str) -> dict:
    """Pinned window if the ledger has one, else derived from tested_date.
    ``required_session`` always comes from tested_date (data coverage)."""
    derived = batch3.derive_holdout_window(tested_date)
    pinned = trial_ledger.get_pinned_holdout_window(ledger_path, family_id)
    if pinned is None:
        return {**derived, "window_source": "derived"}
    return {"holdout_start": pinned["holdout_start"],
            "holdout_end": pinned["holdout_end"],
            "required_session": derived["required_session"],
            "window_source": "pinned"}


# -------------------------------------------------------------- compute

def _signal_factory(module, ticker: str, frames: Mapping) -> Callable:
    def factory(index):
        signal_fn = module.make_signal_fn(ticker, index, frames)
        if not callable(signal_fn):
            raise ValueError(f"{module.FAMILY_ID}: make_signal_fn({ticker}) "
                             "did not return a callable")
        return signal_fn
    return factory


def compute_candidate(spec: dict, module, catalog_entry: dict, frames: Mapping,
                      window: dict, costs: dict,
                      member_errors: dict | None = None) -> dict:
    """Build one item in EXACTLY the shape run_batch3_mining.compute_candidate
    returns (plus batch5-only extras: shape, benchmark, expected_min_trades,
    window_source, per_ticker_windows[private]) so score_batch/evidence see
    nothing new.

    ``member_errors`` extends "drop the member, not the hypothesis" to the
    BACKTEST itself: when a dict is passed (pooled shapes only), a pool
    member whose own backtest raises is recorded there and skipped instead of
    sinking the whole family. Per-member gating that covered only the
    quote gate, the OHLC load and the window buckets would let any other
    single-member failure -- a signal function that raises for one
    instrument, say -- produce a whole-family compute_error with no ledger
    row and no cause in the report: exactly the failure this mechanism
    exists to remove. If EVERY
    member fails, the family legitimately cannot be scored and this raises."""
    family_id, shape = spec["family_id"], spec["shape"]
    traded, signal_symbols = list(spec["traded"]), list(spec["signal"])
    account_type = catalog_entry["account_types"][0]
    tax_rate = cost_tax_inputs.tax_haircut_rate_for(account_type)
    module_frames = {symbol: frames[symbol] for symbol in traded + signal_symbols}
    per_ticker, extra = {}, {}
    # A module may expose its own per-member hygiene counters (how many
    # events it dropped and why). Those decisions shape the scored sample,
    # so they belong in the run output rather than being computed and
    # discarded. Reset first: the sink is module-level state and one process
    # can compute the same family twice.
    reset = getattr(module, "reset_diagnostics", None)
    if callable(reset):
        reset()
    if shape == ROTATION_SHAPE:
        windows = batch3.run_rotation_windows(
            {symbol: frames[symbol] for symbol in traded},
            {symbol: frames[symbol] for symbol in signal_symbols},
            module.make_target_fn(module_frames),
            holdout_start=window["holdout_start"],
            holdout_end=window["holdout_end"],
            cost_bps_by_asset={symbol: costs[symbol]["cost_bps"]
                               for symbol in traded},
            tax_haircut_rate=tax_rate,
            expected_holdout_calendar=batch3._expected_holdout_calendar(
                window["holdout_start"], window["holdout_end"]))
        stats_is, stats_ho = windows["stats_in_sample"], windows["stats_holdout"]
        is_trades, ho_trades = windows["in_sample_trades"], windows["holdout_trades"]
        is_index, ho_index = windows["in_sample_index"], windows["holdout_index"]
        daily = backtest_engine.daily_returns_by_exit_day(is_index, is_trades)
        extra["signal_alignment"] = windows["alignment"]
    else:
        for symbol in list(traded):
            try:
                per_ticker[symbol] = batch3.run_single_asset_windows(
                    frames[symbol],
                    _signal_factory(module, symbol, module_frames),
                    holdout_start=window["holdout_start"],
                    holdout_end=window["holdout_end"],
                    cost_bps=costs[symbol]["cost_bps"],
                    tax_haircut_rate=tax_rate)
            except Exception as exc:
                if member_errors is None:
                    raise
                member_errors[symbol] = f"backtest: {type(exc).__name__}: {exc}"
                traded.remove(symbol)
        if member_errors is not None and not traded:
            raise ValueError(
                f"{family_id}: every pool member's backtest failed: "
                + _render_exclusions(member_errors))
        if member_errors:
            # Never claim a cost for an instrument this run did not trade.
            costs = {symbol: costs[symbol] for symbol in traded}
        if shape in SINGLE_SHAPES:
            one = per_ticker[traded[0]]
            stats_is, stats_ho = one["stats_in_sample"], one["stats_holdout"]
            is_trades, ho_trades = one["in_sample_trades"], one["holdout_trades"]
            is_index, ho_index = one["in_sample_index"], one["holdout_index"]
            daily = backtest_engine.daily_returns_by_exit_day(is_index, is_trades)
        else:
            is_trades = pooling.pool_trades(
                {symbol: per_ticker[symbol]["in_sample_trades"] for symbol in traded})
            ho_trades = pooling.pool_trades(
                {symbol: per_ticker[symbol]["holdout_trades"] for symbol in traded})
            stats_is = pooling.pooled_trade_stats(is_trades)
            stats_ho = pooling.pooled_trade_stats(ho_trades)
            daily = pooling.pooled_daily_returns_by_exit_day(
                {symbol: (per_ticker[symbol]["in_sample_trades"],
                          per_ticker[symbol]["in_sample_index"])
                 for symbol in traded})
            is_index = pd.DatetimeIndex([])
            ho_index = pd.DatetimeIndex([])
            for symbol in traded:
                is_index = is_index.union(per_ticker[symbol]["in_sample_index"])
                ho_index = ho_index.union(per_ticker[symbol]["holdout_index"])
    pooled_diagnostics = getattr(module, "pooled_diagnostics", None)
    if callable(pooled_diagnostics):
        extra["event_diagnostics"] = pooled_diagnostics()
    daily = daily.tail(constants.DAILY_RETURNS_CORRELATION_WINDOW_DAYS)
    tickers_label = pool_ticker_label(traded)
    item = {
        "family_id": family_id, "ticker": tickers_label, "tickers": traded,
        "holdout_start": window["holdout_start"],
        "holdout_end": window["holdout_end"], "holdout_complete": True,
        "stats_in_sample": stats_is, "stats_holdout": stats_ho,
        "holdout_net_edge_bps": stats_ho["net_edge_bps"],
        "daily_returns": daily,
        "account_type": account_type, "tax_haircut_rate": tax_rate,
        "execution_costs": costs,
        "data_through": {symbol: frames[symbol].index.max().strftime("%Y-%m-%d")
                         for symbol in traded + signal_symbols},
        "catalog_parameters": dict(spec["parameters"]),
        "raw_catalog_parameters": dict(catalog_entry["parameters"]),
        "note": (f"batch5 {shape} on {tickers_label}; cost source "
                 f"{costs[traded[0]]['source']}; directional batch"),
        "in_sample_start": (is_index.min().strftime("%Y-%m-%d")
                            if len(is_index) else window["holdout_start"]),
        "in_sample_end": (is_index.max().strftime("%Y-%m-%d")
                          if len(is_index) else window["holdout_start"]),
        # batch5 extras (report-only context; public_result keeps them)
        "shape": shape, "benchmark": spec["benchmark"],
        "signal_tickers": signal_symbols,
        "expected_min_trades": spec["expected_min_trades"],
        "window_source": window["window_source"],
        "snapshots": spec["snapshots"],
        # excluded by public_result
        "in_sample_trades": is_trades, "holdout_trades": ho_trades,
        "in_sample_index": is_index, "holdout_index": ho_index,
        **extra,
    }
    return item, per_ticker


# ---------------------------------------------- directional-support panel

def _aligned(series_by_key: Mapping[str, pd.Series], *, how: str) -> pd.Series:
    union = pd.DatetimeIndex([])
    for series in series_by_key.values():
        union = union.union(pd.DatetimeIndex(series.index))
    union = union.sort_values()
    total = pd.Series(0.0, index=union)
    for series in series_by_key.values():
        total = total.add(series.reindex(union, fill_value=0.0), fill_value=0.0)
    if how == "mean" and len(series_by_key) > 1:
        total = total / float(len(series_by_key))
    return total


def _safe_ols(strategy: pd.Series, benchmark: pd.Series) -> tuple:
    try:
        return benchmark_relative.ols_alpha_beta(strategy, benchmark), None
    except ValueError as exc:
        return None, str(exc)


def _time_matched_excess_bps(trades: pd.DataFrame, bench: pd.DataFrame) -> tuple:
    if trades is None or len(trades) == 0:
        return None, "no trades"
    try:
        excess = benchmark_relative.time_matched_benchmark_excess(trades, bench)
    except ValueError as exc:
        return None, str(exc)
    value = float(np.mean(excess.to_numpy(dtype=float)) * 1e4)
    return (value if math.isfinite(value) else None), None


def build_directional_panel(item: dict, per_ticker: Mapping, frames: Mapping,
                            *, market_frame: pd.DataFrame) -> dict:
    """The per-candidate half of the panel (everything that does not need
    the batch's cross-candidate daily-Sharpe variance). REPORT-ONLY."""
    start = pd.Timestamp(item["holdout_start"])
    traded, shape = list(item["tickers"]), item["shape"]
    costs = item["execution_costs"]
    notes = []

    def _bucket(symbol):
        frame = frames[symbol]
        return frame.loc[frame.index < start]

    sleeves = {}
    if shape == ROTATION_SHAPE:
        trades = item["in_sample_trades"]
        for symbol in traded:
            leg = trades.loc[trades["asset"] == symbol] if len(trades) else trades
            sleeves[symbol] = benchmark_relative.mark_to_market_daily_returns(
                _bucket(symbol), leg, cost_bps=costs[symbol]["cost_bps"])
        strategy_daily = _aligned(sleeves, how="sum")
        notes.append("rotation: per-leg mark-to-market sleeves SUMMED (legs never "
                     "overlap); a switch-day exit is marked through the close "
                     "although the engine exits at the open -- report-only "
                     "approximation")
        own_symbols = [traded[0]]
    else:
        for symbol in traded:
            sleeves[symbol] = benchmark_relative.mark_to_market_daily_returns(
                _bucket(symbol), per_ticker[symbol]["in_sample_trades"],
                cost_bps=costs[symbol]["cost_bps"])
        if len(traded) == 1:
            strategy_daily = sleeves[traded[0]]
        else:
            strategy_daily = _aligned(sleeves, how="mean")
            notes.append("pooled: equal-weight MEAN of per-ticker sleeves on the "
                         "union calendar, 0.0-filled where a ticker has no bar")
        own_symbols = traded
    own_bh = _aligned({symbol: benchmark_relative.buy_and_hold_daily_returns(
        _bucket(symbol)) for symbol in own_symbols}, how="mean")
    if len(own_symbols) > 1:
        notes.append("own buy-and-hold: equal-weight mean of the pool's "
                     "buy-and-hold series on the same union-calendar basis")
    spy_bh = benchmark_relative.buy_and_hold_daily_returns(
        market_frame.loc[market_frame.index < start])
    ols_own, ols_own_error = _safe_ols(strategy_daily, own_bh)
    ols_spy, ols_spy_error = _safe_ols(strategy_daily, spy_bh)
    tm_is, tm_is_error = _time_matched_excess_bps(item["in_sample_trades"], market_frame)
    tm_ho, tm_ho_error = _time_matched_excess_bps(item["holdout_trades"], market_frame)
    moments = benchmark_relative.daily_sharpe_moments(strategy_daily)
    return {
        "report_only": True,
        "strategy_daily_days": int(len(strategy_daily)),
        "own_benchmark": own_symbols,
        "market_benchmark": MARKET_BENCHMARK,
        "daily_sharpe_annualized": benchmark_relative.sharpe_annualized(strategy_daily),
        "daily_moments": moments,
        "ols_vs_own": ols_own, "ols_vs_own_error": ols_own_error,
        "ols_vs_spy": ols_spy, "ols_vs_spy_error": ols_spy_error,
        "full_window_vs_own": benchmark_relative.full_window_summary(
            strategy_daily, own_bh),
        "time_matched_spy_excess_bps": {
            "in_sample": tm_is, "in_sample_error": tm_is_error,
            "holdout": tm_ho, "holdout_error": tm_ho_error},
        "notes": notes,
    }


def finalize_directional_panels(items: list, correction: dict) -> dict:
    """Second pass: the batch's daily-Sharpe variance (ddof=1; falls back to
    constants.DAILY_SR_VARIANCE below 2 finite values or at zero variance),
    then daily DSR + flags on every panel.

    The fallback is deliberately NOT constants.DEFAULT_SR_VARIANCE_FLOOR,
    even though compute_shared_correction applies that constant in the same
    structural position on its own tier. That constant's own comment
    declares it PER-TRADE, and per-trade and daily Sharpe variances live on
    very different scales (often two orders of magnitude apart), which
    moves the DSR bar substantially. Mirroring the per-trade rule onto the
    daily path is a unit swap, not a shared convention --
    account_relative.score_account refuses to score at all rather than
    borrow the per-trade floor, and this is the same refusal expressed as a
    daily constant.
    """
    sharpes = []
    for item in items:
        panel_ = item.get("directional_panel")
        if panel_ is None:
            continue
        value = float(panel_["daily_moments"]["sr"])
        if math.isfinite(value) and panel_["daily_moments"]["n"] >= 2:
            sharpes.append(value)
    variance = float(np.var(sharpes, ddof=1)) if len(sharpes) >= 2 else math.nan
    if math.isfinite(variance) and variance > 0:
        source = f"sample variance (ddof=1) of {len(sharpes)} batch daily Sharpes"
    else:
        variance = float(constants.DAILY_SR_VARIANCE)
        source = ("constants.DAILY_SR_VARIANCE fallback (fewer than 2 "
                  "finite batch daily Sharpes, or zero variance)")
    inputs = {"n_trials": correction["n_trials"], "sr_variance_daily": variance,
              "sr_variance_source": source}
    for item in items:
        panel_ = item.get("directional_panel")
        if panel_ is None:
            continue
        moments = panel_["daily_moments"]
        daily_dsr, dsr_error = None, None
        if moments["n"] < 2:
            dsr_error = "fewer than 2 daily observations"
        else:
            try:
                daily_dsr = float(benchmark_relative.daily_frequency_dsr(
                    sr=moments["sr"], n=moments["n"], skew=moments["skew"],
                    kurtosis=moments["kurtosis"],
                    n_trials=correction["n_trials"], sr_variance=variance))
            except Exception as exc:
                dsr_error = f"{type(exc).__name__}: {exc}"
        ols_own = panel_.get("ols_vs_own")
        t_own = None if ols_own is None else float(ols_own["t_alpha_hac"])
        tm_is = panel_["time_matched_spy_excess_bps"]["in_sample"]
        if t_own is None or math.isnan(t_own):
            beta_in_disguise = directional_support = None
        else:
            beta_in_disguise = bool(t_own <= 0.0)
            directional_support = bool(
                t_own > DIRECTIONAL_SUPPORT_T_ALPHA and tm_is is not None and tm_is > 0.0)
        panel_["daily_dsr"] = daily_dsr
        panel_["daily_dsr_error"] = dsr_error
        panel_["daily_dsr_inputs"] = dict(inputs)
        panel_["flags"] = {"beta_in_disguise": beta_in_disguise,
                           "directional_support": directional_support}
    return inputs


# ---------------------------------------------------------------- scope

def build_scope(*, accepted_path: str, quotes_path: str, catalog_path: str,
                ledger_path: str, modules_dir: str, run_dir: str,
                families: list | None = None,
                retest_family_ids=(), ohlc_loader: Callable | None = None,
                repo_root: str = REPO_ROOT) -> dict:
    """Load and reconcile every static input BEFORE any backtest: accepted
    candidates, quotes, catalog, discovered modules, retest set."""
    accepted_payload = load_accepted(accepted_path)
    quotes_payload = load_quotes(quotes_path)
    entries = catalog.load_catalog(catalog_path)
    catalog_by_id = {entry["family_id"]: entry for entry in entries}
    modules, module_problems = discover_modules(modules_dir, repo_root=repo_root)
    accepted = list(accepted_payload["accepted"])
    if families:
        wanted = set(families)
        unknown = sorted(wanted - {c["family_id"] for c in accepted})
        if unknown:
            raise ValueError(f"--families names non-accepted ids: {unknown}")
        accepted = [c for c in accepted if c["family_id"] in wanted]
    accepted_ids = {c["family_id"] for c in accepted}
    unmatched_modules = sorted(fid for fid in modules if fid not in accepted_ids)
    return {
        "accepted": accepted,
        "accepted_path": accepted_path,
        "accepted_sha256": batch3._file_sha256(accepted_path),
        "quotes": quotes_payload, "quotes_path": quotes_path,
        "catalog_path": catalog_path, "catalog_by_id": catalog_by_id,
        "modules": modules, "modules_dir": os.path.abspath(modules_dir),
        "module_problems": module_problems,
        "unmatched_modules": unmatched_modules,
        "retest_family_ids": frozenset(retest_family_ids or ()),
        "ohlc_loader": ohlc_loader or default_ohlc_loader,
        "run_dir": os.path.abspath(run_dir),
        "families_filter": list(families or []),
    }


def _skip(skipped: list, family_id: str, code: str, reason: str) -> None:
    skipped.append({"family_id": family_id, "reason_code": code, "reason": reason})


POOL_LABEL_MAX_CHARS = 96

# PROVISIONAL_DEFAULT -- the one invented number here, same
# convention as MIN_POOLED_SR_VARIANCE_SAMPLES and NEAR_MISS_DSR_MARGIN.
# Per-member exclusion protects a pooled family from ONE unusable
# instrument; without a coverage floor it could score a full pre-registered
# verdict on a small fraction of its declared pool: a quote file covering
# only a few hundred symbols could leave a large pooled family scoring on a
# handful of members and recording that as the declared hypothesis. Worse,
# quote-gate exclusion is not return-neutral in general (spread width can
# correlate with a member's own realized edge), and that selection tilt is
# largest exactly in a tiny surviving subset. A materially narrower pool
# is a DIFFERENT hypothesis, so
# the run refuses rather than quietly relabels it. A refusal charges no
# trial, so the family stays eligible for a later run with better data.
MIN_POOL_COVERAGE = 0.80


def _pool_coverage_ok(kept, declared) -> bool:
    if not declared:
        return False
    return (len(kept) / len(declared)) >= MIN_POOL_COVERAGE


def pool_ticker_label(traded: list) -> str:
    """A short label for a family's traded set, safe as a FILENAME and as an
    append-only ledger row.

    Joining every member of a pool of several hundred tickers yields a label
    thousands of characters long; fed through ``reporting.brief_filename``
    that would exceed the filesystem's filename limit and the write would
    raise OSError -- AFTER score_batch has pinned the outcome, so a surviving
    pooled family would crash the run and then be refused on re-run as
    already pinned. It would fire only on a SURVIVOR, the one outcome that
    matters most, which is why the joined label is bounded here
    rather than at each consumer. A short pool keeps its readable label; the
    full member set is always carried in ``tickers``, ``pool_declared`` and
    ``pool_excluded``, and named in the run report."""
    joined = "/".join(traded)
    if len(joined) <= POOL_LABEL_MAX_CHARS:
        return joined
    return f"pool-of-{len(traded)}"


def _render_exclusions(excluded: Mapping) -> str:
    return "; ".join(f"{symbol}: {reason}"
                     for symbol, reason in sorted(excluded.items()))


def compute_batch(scope: dict, *, tested_date: str, ledger_path: str) -> dict:
    """Preflight every candidate (module, drift, costs, window), load every
    OHLC frame once, then compute all feasible candidates. No backtest runs
    before every input has been loaded and checked."""
    skipped, plans = [], []
    quotes_payload = scope["quotes"]
    problem_by_module = {p["module"]: p["reason"] for p in scope["module_problems"]}
    for candidate in scope["accepted"]:
        family_id = candidate["family_id"]
        found = scope["modules"].get(family_id)
        if found is None:
            reason = problem_by_module.get(family_id)
            if reason:
                _skip(skipped, family_id, "invalid_module", reason)
            else:
                _skip(skipped, family_id, "no_module",
                      f"no valid module for {family_id} under "
                      f"{portable_path(scope['modules_dir'])}")
            continue
        spec, module = found["spec"], found["module"]
        entry = scope["catalog_by_id"].get(family_id)
        if entry is None:
            _skip(skipped, family_id, "not_in_catalog",
                  "family_id absent from the catalog (run add_batch5_to_catalog.py first)")
            continue
        if entry.get("status") != "eligible":
            _skip(skipped, family_id, "catalog_status",
                  f"catalog status is {entry.get('status')!r}, not eligible")
            continue
        drift = parameter_drift(spec["parameters"], entry.get("parameters"))
        if drift is not None:
            _skip(skipped, family_id, "parameter_drift", drift)
            continue
        # A POOLED family is a pool: one unusable member costs the pipeline
        # that member, not the hypothesis. A single or rotation candidate has
        # no pool to fall back on, so one bad leg still skips the family.
        pooled = spec["shape"] in POOLED_SHAPES
        declared = list(spec["traded"])
        excluded = {}
        if pooled:
            costs = {}
            for symbol in declared:
                try:
                    costs.update(execution_costs_for([symbol], quotes_payload))
                except ValueError as exc:
                    excluded[symbol] = f"quote gate: {exc}"
            if not costs:
                _skip(skipped, family_id, "missing_data",
                      "every pool member excluded at the quote gate: "
                      + _render_exclusions(excluded))
                continue
        else:
            try:
                costs = execution_costs_for(declared, quotes_payload)
            except ValueError as exc:
                _skip(skipped, family_id, "missing_data", f"quote gate: {exc}")
                continue
        if (trial_ledger.has_been_tested(ledger_path, family_id)
                and family_id not in scope["retest_family_ids"]
                and trial_ledger.latest_outcome(ledger_path, family_id)
                != "not_evaluated_error"):
            _skip(skipped, family_id, "already_pinned_not_in_retest",
                  "already first-tested with a real outcome and not listed in "
                  "--retest; refusing to re-score without an explicit retest request")
            continue
        try:
            window = candidate_window(ledger_path, family_id, tested_date)
        except Exception as exc:
            _skip(skipped, family_id, "window_error", f"{type(exc).__name__}: {exc}")
            continue
        plans.append({"family_id": family_id, "spec": spec, "module": module,
                      "entry": entry, "costs": costs, "window": window,
                      "pooled": pooled, "pool_declared": declared,
                      "pool_excluded": excluded,
                      "effective": [s for s in declared if s in costs]})

    symbols = {MARKET_BENCHMARK}
    for plan in plans:
        symbols.update(plan["effective"])
        symbols.update(plan["spec"]["signal"])
        symbols.add(plan["spec"]["benchmark"])
    frames, load_errors, coverage = {}, {}, {}
    cache_dir = os.path.join(scope["run_dir"], "artifacts", "ohlc")
    required_by_family = {plan["family_id"]: pd.Timestamp(plan["window"]["required_session"])
                          for plan in plans}
    strictest = max(required_by_family.values()) if required_by_family else None
    for symbol in sorted(symbols):
        try:
            report = {}
            frame, origin = load_frame(symbol, ohlc_loader=scope["ohlc_loader"],
                                       cache_dir=cache_dir, report=report)
            frames[symbol] = frame
            coverage[symbol] = {"first": frame.index.min().strftime("%Y-%m-%d"),
                                "last": frame.index.max().strftime("%Y-%m-%d"),
                                "origin": origin, **report}
            if strictest is not None and frame.index.max() < strictest:
                load_errors[symbol] = (
                    f"data ends {frame.index.max().date()}, before required "
                    f"session {strictest.date()}")
        except Exception as exc:
            load_errors[symbol] = f"{type(exc).__name__}: {exc}"

    computed, per_ticker_by_family, panel_errors = [], {}, {}
    market_frame = frames.get(MARKET_BENCHMARK)
    for plan in plans:
        family_id, spec = plan["family_id"], plan["spec"]
        pooled, excluded = plan["pooled"], plan["pool_excluded"]
        effective = list(plan["effective"])
        # Signal-only and benchmark series are never pool members: they carry
        # the whole family, so a failure there is still fatal.
        support = dict.fromkeys(list(spec["signal"])
                                + [spec["benchmark"], MARKET_BENCHMARK])
        support_problems = [f"{symbol}: {load_errors[symbol]}"
                            for symbol in support if symbol in load_errors]
        if support_problems:
            _skip(skipped, family_id, "missing_data", "; ".join(support_problems))
            continue
        if pooled:
            for symbol in list(effective):
                if symbol in load_errors:
                    excluded[symbol] = f"ohlc: {load_errors[symbol]}"
                    effective.remove(symbol)
            # A recent listing with no pre-holdout history makes its bucket empty, and
            # batch3._require_bucket then raises for the WHOLE family. Ask the
            # engine's own mask helper, so this guard cannot drift from the
            # rule it is anticipating.
            for symbol in list(effective):
                index = frames[symbol].index
                is_mask, hold_mask = batch3._window_masks(
                    index, plan["window"]["holdout_start"],
                    plan["window"]["holdout_end"])
                n_is, n_hold = int(is_mask.sum()), int(hold_mask.sum())
                if n_is < 2 or n_hold < 2:
                    excluded[symbol] = (
                        f"insufficient history: fewer than two bars in a "
                        f"window bucket (in_sample {n_is}, holdout {n_hold}; "
                        f"first bar {index.min().strftime('%Y-%m-%d')})")
                    effective.remove(symbol)
            if not effective:
                _skip(skipped, family_id, "missing_data",
                      "every pool member excluded before any backtest: "
                      + _render_exclusions(excluded))
                continue
        else:
            member_problems = [f"{symbol}: {load_errors[symbol]}"
                               for symbol in effective if symbol in load_errors]
            if member_problems:
                _skip(skipped, family_id, "missing_data", "; ".join(member_problems))
                continue
        if pooled and not _pool_coverage_ok(effective, plan["pool_declared"]):
            kept, declared = len(effective), len(plan["pool_declared"])
            _skip(skipped, family_id, "pool_coverage",
                  f"only {kept} of {declared} declared pool members are "
                  f"usable ({kept / declared:.1%}), below the "
                  f"{MIN_POOL_COVERAGE:.0%} floor: scoring this would record a "
                  f"pre-registered verdict for a materially narrower "
                  f"hypothesis than the one registered. Excluded: "
                  + _render_exclusions(excluded))
            continue
        run_spec = {**spec, "traded": effective}
        costs = {symbol: plan["costs"][symbol] for symbol in effective}
        member_errors = {} if pooled else None
        try:
            item, per_ticker = compute_candidate(
                run_spec, plan["module"], plan["entry"], frames, plan["window"],
                costs, member_errors=member_errors)
        except Exception as exc:
            # Carry the exclusions already recorded into the skip reason:
            # they are the context for WHY the remainder failed, and
            # pool_exclusions is built from scored items only, so otherwise
            # they vanish from every output.
            detail = f"{type(exc).__name__}: {exc}"
            if excluded:
                detail += "; already excluded: " + _render_exclusions(excluded)
            _skip(skipped, family_id, "compute_error", detail)
            continue
        if pooled:
            excluded.update(member_errors or {})
            if not _pool_coverage_ok(item["tickers"], plan["pool_declared"]):
                kept, declared = len(item["tickers"]), len(plan["pool_declared"])
                _skip(skipped, family_id, "pool_coverage",
                      f"only {kept} of {declared} declared pool members "
                      f"survived the backtest ({kept / declared:.1%}), below "
                      f"the {MIN_POOL_COVERAGE:.0%} floor. Excluded: "
                      + _render_exclusions(excluded))
                continue
            item["pool_declared"] = plan["pool_declared"]
            item["pool_excluded"] = dict(excluded)
        try:
            item["directional_panel"] = build_directional_panel(
                item, per_ticker, frames, market_frame=market_frame)
        except Exception as exc:
            # The panel is report-only; its failure must never cost the
            # candidate its pre-registered score.
            item["directional_panel"] = None
            panel_errors[family_id] = f"{type(exc).__name__}: {exc}"
        computed.append(item)
        per_ticker_by_family[family_id] = per_ticker
    return {"computed": computed, "skipped": skipped,
            "per_ticker": per_ticker_by_family,
            "input_coverage": coverage, "input_errors": load_errors,
            "panel_errors": panel_errors,
            "pool_exclusions": {item["family_id"]: item["pool_excluded"]
                                for item in computed
                                if item.get("pool_excluded")},
            "windows": {plan["family_id"]: plan["window"] for plan in plans}}


# -------------------------------------------------------------- outputs

def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)  # 'inf' / '-inf' / 'nan' -- json allow_nan=False
    if isinstance(value, (pd.Series, pd.DataFrame, pd.Index)):
        raise TypeError("pandas objects must be stripped before serialization")
    return value


def _fmt(value, digits=3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return repr(number)
    return f"{number:.{digits}f}"


def _report_safe(text: str) -> str:
    # results_log's missing-data splitter breaks on ", <slug>: " -- keep a
    # reason on one item by never emitting ", " inside it.
    return str(text).replace("\r", " ").replace("\n", " ").replace(", ", "; ")


def render_candidate_table(scored: list) -> str:
    header = ("| family | shape | tickers | trades IS/HO | per-trade SR | DSR | "
              "stage | daily SR ann | daily DSR | t_alpha own | t_alpha SPY | "
              "TM-SPY excess bps IS/HO | flags |")
    rule = "|" + "---|" * 13
    rows = [header, rule]
    for item in sorted(scored, key=lambda i: i["family_id"]):
        panel_ = item.get("directional_panel") or {}
        ols_own = panel_.get("ols_vs_own") or {}
        ols_spy = panel_.get("ols_vs_spy") or {}
        tm = panel_.get("time_matched_spy_excess_bps") or {}
        flags = panel_.get("flags") or {}
        flag_text = ", ".join(name for name, on in flags.items() if on) or "none"
        if item.get("pool_declared"):
            # A pool of hundreds of names cannot be printed inline; the members kept and
            # the reason for every drop are in the Pool exclusions section.
            tickers_cell = f"pool {len(item['tickers'])}/{len(item['pool_declared'])}"
        else:
            tickers_cell = item["ticker"]
        rows.append(
            f"| {item['family_id']} | {item.get('shape', '?')} | {tickers_cell} | "
            f"{item['stats_in_sample']['trade_count']}/"
            f"{item['stats_holdout']['trade_count']} | "
            f"{_fmt(item['stats_in_sample']['sharpe'])} | "
            f"{_fmt(item.get('dsr_value'), 4)} | {item['stage']} | "
            f"{_fmt(panel_.get('daily_sharpe_annualized'))} | "
            f"{_fmt(panel_.get('daily_dsr'), 4)} | "
            f"{_fmt(ols_own.get('t_alpha_hac'))} | {_fmt(ols_spy.get('t_alpha_hac'))} | "
            f"{_fmt(tm.get('in_sample'), 1)}/{_fmt(tm.get('holdout'), 1)} | "
            f"{flag_text} |")
    return "\n".join(rows) + "\n"


def render_report(scored: list, skipped: list, correction: dict, *,
                  tested_date: str, n_considered: int, excluded_ruled_out: list,
                  panel_inputs: dict, provenance: dict, dry_run: bool) -> str:
    survivors = [item for item in scored if item["stage"] == "survived"]
    passed_dsr = [item for item in scored
                  if item["stage"] in ("survived", "cut_holdout")]
    missing = [f"{row['family_id']}: {_report_safe(row['reason'])}" for row in skipped]
    text = reporting.render_run_report(
        date=tested_date, n_considered=n_considered,
        excluded_ruled_out=excluded_ruled_out, excluded_missing_data=missing,
        not_attempted_budget=[], n_passed_dsr=len(passed_dsr),
        n_survivors=len(survivors),
        survivor_briefs=[f"{item['family_id']} ({item['ticker']})"
                         for item in survivors],
        budget_used="n/a (script-driven)", budget_ceiling="n/a",
        n_families_tested_this_run=len(scored),
        n_families_correction=float(correction["n_trials"]))
    parts = [text, "\n## Batch5 candidate table\n\n"]
    if dry_run:
        parts.append("DRY RUN -- no ledger record was written.\n\n")
    parts.append("Pre-registered columns (per-trade SR, DSR, stage) are the "
                 "recorded score. Every column from `daily SR ann` rightward is "
                 "the REPORT-ONLY directional-support panel and influences no "
                 "stage, DSR or ledger record.\n\n")
    parts.append(render_candidate_table(scored))
    parts.append(f"\nDaily-DSR inputs: n_trials {correction['n_trials']:.2f}; "
                 f"sr_variance_daily {panel_inputs['sr_variance_daily']:.6g} "
                 f"({panel_inputs['sr_variance_source']}).\n")
    parts.append(f"Shared correction: n_this {correction['n_this']:.2f}, "
                 f"post-batch ledger floor {correction['post_batch_ledger_floor']}, "
                 f"sr_variance (per-trade) {correction['sr_variance']:.6g}.\n")
    # EVERY pooled family is listed, including one that dropped nothing:
    # the table cell reads "pool KEPT/DECLARED" rather than the instruments,
    # so without this section a reader of the report alone could not
    # reconstruct which instruments a pooled family actually traded.
    pooled_items = [item for item in scored if item.get("pool_declared")]
    if pooled_items:
        parts.append("\n## Pooled families and their members\n\n")
        parts.append("A pooled family drops an individual member it cannot "
                     "price or backtest and scores on the remainder; the "
                     "family is still ONE trial. Every drop is named here, "
                     "with its reason.\n\n")
        for item in sorted(pooled_items, key=lambda i: i["family_id"]):
            kept, declared = len(item["tickers"]), len(item["pool_declared"])
            excluded = item.get("pool_excluded") or {}
            parts.append(f"- {item['family_id']}: kept {kept} of {declared} "
                         f"declared members\n")
            members = "/".join(item["tickers"])
            if len(members) <= POOL_LABEL_MAX_CHARS:
                parts.append(f"  - tested: {members}\n")
            else:
                parts.append(f"  - tested: {kept} instruments, listed in "
                             f"batch5-results.json under `tickers`\n")
            for symbol, reason in sorted(excluded.items()):
                parts.append(f"  - excluded {symbol}: {_report_safe(reason)}\n")
            if not excluded:
                parts.append("  - excluded: none\n")
    parts.append("\n## Skipped candidates\n\n")
    if skipped:
        for row in sorted(skipped, key=lambda r: r["family_id"]):
            parts.append(f"- {row['family_id']} [{row['reason_code']}]: "
                         f"{_report_safe(row['reason'])}\n")
    else:
        parts.append("- none\n")
    parts.append("\n## Input provenance\n\n")
    for key, value in provenance.items():
        parts.append(f"- {key}: {value}\n")
    return "".join(parts)


def _brief_ticker_slug(ticker: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "+", ticker)


def render_briefs(scored: list, catalog_by_id: dict, correction: dict) -> dict:
    briefs = {}
    for item in scored:
        if item["stage"] != "survived":
            continue
        entry = catalog_by_id[item["family_id"]]
        stats = item["stats_in_sample"]
        panel_ = item.get("directional_panel") or {}
        flags = panel_.get("flags") or {}
        text = reporting.render_candidate_brief(
            family_name=entry["name"], ticker=item["ticker"],
            mechanism_hypothesis=entry["mechanism"],
            parameters_tested=_canonical(item["catalog_parameters"]),
            cost_source=item["execution_costs"][item["tickers"][0]]["source"],
            tax_treatment=(f"{item['account_type']}: haircut rate "
                           f"{item['tax_haircut_rate']}"),
            holdout_start_date=item["holdout_start"],
            dsr_value=float(item["dsr_value"]),
            n_families_correction=float(correction["n_trials"]),
            trade_count=stats["trade_count"],
            other_summary_stats=(
                f"net edge {stats['net_edge_bps']:.1f} bps/trade, per-trade "
                f"Sharpe {stats['sharpe']:.3f}, skew {stats['skew']:.3f}, "
                f"kurtosis {stats['kurtosis']:.3f}"),
            holdout_window=f"{item['holdout_start']} to {item['holdout_end']}",
            holdout_result=(
                f"{item['stats_holdout']['trade_count']} trades, net edge "
                f"{item['holdout_net_edge_bps']:.1f} bps/trade"),
            survival_case=(
                "REPORT-ONLY directional-support panel: "
                f"t_alpha_hac vs own B&H {_fmt((panel_.get('ols_vs_own') or {}).get('t_alpha_hac'))}, "
                f"vs SPY {_fmt((panel_.get('ols_vs_spy') or {}).get('t_alpha_hac'))}; "
                f"time-matched SPY excess IS/HO "
                f"{_fmt((panel_.get('time_matched_spy_excess_bps') or {}).get('in_sample'), 1)}/"
                f"{_fmt((panel_.get('time_matched_spy_excess_bps') or {}).get('holdout'), 1)} bps; "
                f"daily DSR {_fmt(panel_.get('daily_dsr'), 4)}; flags "
                f"beta_in_disguise={flags.get('beta_in_disguise')}, "
                f"directional_support={flags.get('directional_support')}. "
                "A survivor still needs the 12-window shift test and its own "
                "pre-registered gate before any capital."),
            account_eligibility_notes=", ".join(entry.get("account_types", [])))
        briefs[reporting.brief_filename(
            item["family_id"], _brief_ticker_slug(item["ticker"]))] = text
    return briefs


def write_trade_csvs(scored: list, run_dir: str) -> list:
    trades_dir = os.path.join(run_dir, "artifacts", "trades")
    os.makedirs(trades_dir, exist_ok=True)
    written = []
    for item in scored:
        for bucket in ("in_sample", "holdout"):
            path = os.path.join(trades_dir, f"{item['family_id']}_{bucket}.csv")
            item[f"{bucket}_trades"].to_csv(path, index=False)
            written.append(path)
    return written


def run_batch(scope: dict, *, tested_date: str, run_id: str,
              ledger_path: str, dry_run: bool) -> dict:
    """The ``batch_runner`` handed to run_batch3_mining.execute_batch_run."""
    refuse_synthetic_quotes(scope["quotes"], dry_run=dry_run)
    batch = compute_batch(scope, tested_date=tested_date, ledger_path=ledger_path)
    computed = batch["computed"]
    correction = batch3.compute_shared_correction(computed, ledger_path)
    panel_inputs = finalize_directional_panels(computed, correction)
    scored = batch3.score_batch(
        computed, ledger_path=ledger_path, run_id=run_id,
        tested_date=tested_date, dry_run=dry_run, correction=correction,
        retest_family_ids=scope["retest_family_ids"])
    run_dir = scope["run_dir"]
    spy_quote = scope["quotes"]["quotes"].get(MARKET_BENCHMARK)
    provenance = {
        "accepted_path": portable_path(scope["accepted_path"]),
        "accepted_sha256": scope["accepted_sha256"],
        "quotes_path": portable_path(scope["quotes_path"]),
        "quotes_sampled_at_local": scope["quotes"]["sampled_at_local"],
        "quotes_market_state": scope["quotes"]["market_state"],
        "quotes_source": scope["quotes"].get("source"),
        "spy_quote_cross_check": _json_safe(spy_quote) if spy_quote else None,
        "modules_dir": portable_path(scope["modules_dir"]),
        "catalog_path": portable_path(scope["catalog_path"]),
        "ledger_path": portable_path(ledger_path),
        "families_filter": scope["families_filter"],
        "retest_family_ids": sorted(scope["retest_family_ids"]),
    }
    excluded_ruled_out = [row["family_id"] for row in batch["skipped"]
                          if row["reason_code"] == "catalog_status"]
    report = render_report(
        scored, batch["skipped"], correction, tested_date=tested_date,
        n_considered=len(scope["accepted"]), excluded_ruled_out=excluded_ruled_out,
        panel_inputs=panel_inputs, provenance=provenance, dry_run=dry_run)
    briefs = render_briefs(scored, scope["catalog_by_id"], correction)
    if dry_run:
        # A dry run's report must NOT be named run-report.md: results_log
        # globs <research_root>/*/run-report.md as the record of REAL runs.
        report_path = os.path.join(run_dir, "run-report-dry-run.md")
        batch3._atomic_write_text(report_path, report)
        written = [report_path]
        for filename, text in briefs.items():
            path = os.path.join(run_dir, "dry-run-" + filename)
            batch3._atomic_write_text(path, text)
            written.append(path)
    else:
        written = reporting.write_run_outputs(run_dir, report, briefs)
    written += write_trade_csvs(scored, run_dir)
    results = []
    for item in scored:
        public = batch3.public_result(item)
        results.append(_json_safe(public))
    return {
        "schema_version": 1, "batch": "batch5", "run_id": run_id,
        "tested_date": tested_date, "dry_run": dry_run,
        "shared_correction": _json_safe(correction),
        "directional_panel_inputs": _json_safe(panel_inputs),
        "results": results,
        "skipped": batch["skipped"],
        "module_problems": scope["module_problems"],
        "unmatched_modules": scope["unmatched_modules"],
        "windows": batch["windows"],
        "panel_errors": batch["panel_errors"],
        "pool_exclusions": batch["pool_exclusions"],
        "input_coverage": batch["input_coverage"],
        "input_errors": batch["input_errors"],
        "provenance": provenance,
        "outputs_written": [portable_path(path) for path in written],
    }


def _split_ids(raw: str | None) -> list:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--accepted", default=DEFAULT_ACCEPTED)
    parser.add_argument("--quotes", default=DEFAULT_QUOTES)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--tested-date", default=DEFAULT_TESTED_DATE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--families", default=None,
                        help="comma-separated subset of accepted family_ids")
    parser.add_argument("--retest", default=None,
                        help="comma-separated family_ids allowed to re-score "
                             "against their pinned window (no new trial charge)")
    parser.add_argument("--ledger", default=LEDGER_PATH)
    parser.add_argument("--catalog", default=CATALOG_PATH)
    parser.add_argument("--modules-dir", default=DEFAULT_MODULES_DIR)
    parser.add_argument("--research-root", default=RESEARCH_ROOT)
    parser.add_argument("--results-log-out", default=RESULTS_LOG_PATH)
    parser.add_argument("--near-miss-log-out", default=NEAR_MISS_LOG_PATH)
    parser.add_argument("--out", default=None,
                        help="results JSON path (default <research-root>/<run-id>/"
                             "batch5-results.json, or batch5-dry-run.json)")
    args = parser.parse_args(argv)
    run_dir = os.path.join(args.research_root, args.run_id)
    output_path = args.out or os.path.join(
        run_dir, "batch5-dry-run.json" if args.dry_run else "batch5-results.json")
    scope = build_scope(
        accepted_path=args.accepted, quotes_path=args.quotes,
        catalog_path=args.catalog, ledger_path=args.ledger,
        modules_dir=args.modules_dir, run_dir=run_dir,
        families=_split_ids(args.families),
        retest_family_ids=_split_ids(args.retest))
    payload = batch3.execute_batch_run(
        scope, tested_date=args.tested_date, run_id=args.run_id,
        ledger_path=args.ledger, dry_run=args.dry_run,
        output_path=output_path, catalog_path=args.catalog,
        research_root=args.research_root,
        results_log_path=args.results_log_out,
        near_miss_log_path=args.near_miss_log_out,
        batch_runner=run_batch)
    print(json.dumps({
        "output": output_path.replace(os.sep, "/"),
        "tested": len(payload["results"]),
        "skipped": len(payload["skipped"]),
        "survivors": sum(item["stage"] == "survived" for item in payload["results"]),
        "correction_n": payload["shared_correction"]["n_trials"],
        "dry_run": args.dry_run,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
