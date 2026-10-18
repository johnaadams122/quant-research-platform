"""Bespoke ABI-1 driver for intraday_rsi2_reversion_daytrade -- SPY,
long side only, scored under the ABI-1 account-relative construction.

## An older per-trade outcome does not block ABI-1

A family may carry an older ``construction="per_trade"`` outcome. That row
does NOT block an ABI-1 attempt: the one-shot re-run guard keys on
``construction == "ABI-1"``.

## Scope: the LONG side only -- and a real run must say so explicitly

The catalog mechanism describes a mirrored SHORT entry (RSI(2) above 90). The
signal factory ``day_trading_signals.make_intraday_rsi2_reversion_daytrade_
signal_fn`` implements ONLY the long (oversold) side -- its own docstring
says so, and the engine's ``direction`` is a single whole-call parameter --
and this driver scores that same factory through ABI-1. The short side is NOT tested here; a
"no edge" verdict from this driver is a verdict on the long side only, and
every result and ledger note this driver emits says so.

Because a real run spends the family's ONE permanent ABI-1 slot on that
half-definition (a short side would need its own family_id), a real run
refuses to start without ``--accept-long-side-only``: an explicit,
deliberate confirmation, not a default.

## Why a bespoke driver, not the generic dispatch

``scripts/run_abi1_mining.py`` has no intraday-timeframe path (its OHLC
loader fetches DAILY bars only) -- see ``run_vwap_cross_decaying_conviction_
abi1.py``'s module docstring for the full trace; not re-derived here. This
driver is that sibling's single-ticker pattern, reusing:

* ``mining.day_trading_abi1_bridge.build_day_trading_curve`` -- the real
  per-trade ``trades_df`` -> ABI-1 daily account-curve aggregation;
* ``run_batch5_mining.execution_costs_for`` / ``refuse_synthetic_quotes`` /
  ``parameter_drift`` -- the pipeline's own quote-priced cost model and guards;
* ``abi1_driver.evaluate_family`` / ``record_abi1_outcome`` -- the scoring and
  the atomic one-shot ledger write (``trial_ledger.record_outcome_if_not_
  scored`` underneath; a concurrent loser gets ``already_scored_abi1``).

## Parameters are pinned EXPLICITLY (implicit callee defaults defeat drift
## protection)

``PARAMETERS`` is byte-for-byte the catalog entry's ``parameters`` and is
checked against it (``parameter_drift``) before anything is fetched. The
signal factory has one parameter the catalog has no key for -- ``sma_period``,
which the catalog encodes only inside ``exit_rule``'s name
("5bar_sma_cross_or_force_flat"). Letting it fall through to the factory's
own default would be implicit. Here it is DERIVED from the registered
``exit_rule`` (``sma_period_from_exit_rule``) and every factory parameter is
passed as a keyword; ``_registration_drift`` additionally refuses to run if
the factory ever accepts a parameter this driver does not pin, so a new
defaulted knob cannot silently change what gets scored.

## Data range and the one-shot real run

1-minute RTH bars via ``intraday_data.fetch_intraday_bars`` (Alpaca SIP,
cached under ``data/mining/intraday_cache``), as the loader delivers them:
non-empty per session, NOT guaranteed complete. The vendor's SPY minute
history starts at 2016-01-04 (earlier sessions raise
``IntradayDataGapError``), so ``REAL_DATA_START`` is pinned to that floor:
roughly a decade, not the multi-decade daily corpus other families are
scored on. A small number of sessions have missing minutes, and early-close
half days have 210 bars instead of 390. Same-day data is avoided (a real
run ends at the most recent session strictly before today).

The signal factory locates 5-minute bucket ends by COUNTING observed rows
from a session's first bar, and its entry guard only knows the 15:55
force-flat clock. Three consequences are real, causal (no look-ahead) and
disclosed as known limitations (the factory is shared and out of scope for this driver):
after a missing minute every later bucket boundary in that session runs a
minute late; an oversold last bucket of an early-close half day fills at the
NEXT session's 09:30 open; an oversold 15:54 bucket end fills at the 15:55
open and is force-flattened on that same bar. Every result and the ledger
note carry all three counts.

A real (non-``--dry-run``) run spends this family's ONE permanent ABI-1
scoring attempt and needs a REAL open-market quote sample for
**SPY only** (SPY is both the traded instrument and the ABI-1 benchmark),
sampled on the tested date. ``--dry-run`` accepts a quotes file carrying the
literal ``SYNTHETIC`` marker; a real run rejects that marker AND any quote
whose ``market_state`` is not open or whose ``sampled_at_local`` date is not
``--tested-date``. ``--tested-date`` is required on a real run (never
defaulted) and must be TODAY (ET), so an old quote cannot be paired with its
own old date. The run reads the clock ONCE, so the today-rule and the fetch
cutoff always agree. ``--start-date``/``--end-date`` are dry-run-only window
overrides, and an ``--end-date`` of today or later is refused even on a dry
run: the minute cache trusts any existing day file forever, so a
still-forming session would be cached as a partial day.

The daily SPY benchmark comes from ``batch5.load_frame``, which REUSES any
cached ``SPY.csv`` it finds -- and yfinance's newest daily row is an
in-progress bar when fetched during the session. A dry run therefore uses
``<research-root>/<run-id>/artifacts/ohlc``, while a real run gets its own
``.../artifacts/ohlc-<tested-date>`` and never reads a cache left by another
day; rows dated today are dropped from the benchmark whatever the cache held.
The driver refuses (before the slow backtest, deleting the cached file so a
retry re-fetches) when the benchmark ends before the last intraday session.

The result and the permanent ledger note carry: the three counts
above, the last intraday bar's timestamp (a partial final
session shows up as a last bar well before the close), and the identity of
the code that produced the score (this driver's path and sha256 prefix, and
the signal module's) because ``record_abi1_outcome`` keeps no code
attribution for scored rows. Every invocation also mints a unique
``attempt=<token>`` persisted in the ledger row (the note for scored rows, the
evidence's source path for unscored ones), which is how a post-failure ledger
re-read tells its OWN row from a concurrent winner's, even when two runs share
a run id and date.

The ledger row is written BEFORE the results JSON, and ``record_abi1_outcome``
appends the outcome BEFORE registering the first_test row. If either later
step fails the slot is already spent, so the failure is never allowed to eat
the result: a failed results write prints the full result to stdout and exits
1; a failed first_test write returns the result with ``ledger_post_write_error``
set (``main`` writes and prints it, then exits 1); an unreadable ledger is
reported as "ledger state unknown". If no outcome row exists the failure just
propagates and nothing was spent. A frame with naive timestamps is read as New
York time (the loader always delivers tz-aware).

## Cost of a full-history run

A full-history dry run takes on the order of a quarter of an hour on a
warm minute cache. The single-threaded per-bar backtest dominates; the
fetch (cached minute files), the bridge and the ABI-1 evaluation itself
are small by comparison.

Real run (from the repo root, any interpreter with the requirements installed)::

    python scripts/run_intraday_rsi2_reversion_abi1.py \
        --quotes <fresh open-market SPY quotes json> \
        --tested-date YYYY-MM-DD --accept-long-side-only

Never modifies ``day_trading_signals.py``, ``intraday_backtest_engine.py``,
``intraday_data.py``, ``abi1_driver.py`` or the bridge; it only calls them.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
import time
import uuid

import pandas as pd
import pandas_market_calendars as mcal

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
SCRIPTS_ROOT = os.path.join(REPO_ROOT, "scripts")
for _path in (SRC_ROOT, SCRIPTS_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_batch3_mining as batch3
import run_batch5_mining as batch5
from mining import abi1_driver, catalog as catalog_mod, evidence, intraday_data, trial_ledger
from mining import day_trading_abi1_bridge as bridge
from mining import day_trading_signals as mod
from mining.intraday_backtest_engine import run_intraday_backtest

FAMILY_ID = "intraday_rsi2_reversion_daytrade"
TRADED_TICKER = "SPY"
TRADED = (TRADED_TICKER,)
SHAPE = "single_long"
DIRECTION = "long"  # the factory implements the long side only -- see docstring.

# Byte-for-byte the catalog's own "parameters" field (data/mining/catalog.json);
# checked against it by _registration_drift before any fetch. Note the catalog
# spells the RSI levels as integers -- parameter_drift compares canonical JSON,
# so 10 and 10.0 are NOT interchangeable here.
PARAMETERS = {
    "rsi_period": 2,
    "bar_minutes": 5,
    "oversold_level": 10,
    "overbought_level": 90,
    "earliest_entry_et": "10:00",
    "exit_rule": "5bar_sma_cross_or_force_flat",
    "force_flat_et": "15:55",
    "max_hold_days": 1,
}

_EXIT_RULE_PATTERN = re.compile(r"(\d+)bar_sma_cross_or_force_flat")


def sma_period_from_exit_rule(exit_rule: str) -> int:
    """The catalog registers the exit's SMA length only inside the exit_rule
    NAME ("5bar_sma_cross_or_force_flat"); derive it from there rather than
    keeping a second, driftable copy of the number."""
    match = _EXIT_RULE_PATTERN.fullmatch(exit_rule)
    if match is None:
        raise ValueError(
            f"exit_rule {exit_rule!r} is not '<N>bar_sma_cross_or_force_flat' -- "
            "cannot derive the SMA length the signal factory needs")
    return int(match.group(1))


SMA_PERIOD = sma_period_from_exit_rule(PARAMETERS["exit_rule"])

# EVERY keyword the signal factory accepts, passed explicitly on every run.
FACTORY_KWARGS = {
    "rsi_period": PARAMETERS["rsi_period"],
    "bar_minutes": PARAMETERS["bar_minutes"],
    "oversold_level": PARAMETERS["oversold_level"],
    "overbought_level": PARAMETERS["overbought_level"],
    "earliest_entry_et": PARAMETERS["earliest_entry_et"],
    "force_flat_et": PARAMETERS["force_flat_et"],
    "sma_period": SMA_PERIOD,
}

# The vendor's SPY minute history starts here (see docstring).
REAL_DATA_START = "2016-01-04"

CASH_CONVENTION = "sweep_cash_daily"
MARKET_BENCHMARK = "SPY"
TAX_POLICY = "realization"  # single_long shape: 100% of capital in one
                            # position at a time, same as the vwap sibling.
# Pinned, and checked against the catalog's `account_types` by
# _registration_drift: the tax treatment of a one-shot score must not move
# because a catalog list was reordered.
ACCOUNT_TYPE = "taxable"
ACCOUNT_TYPES = (ACCOUNT_TYPE,)

DEFAULT_RUN_ID = "intraday-rsi2-reversion-abi1-v1"
LEDGER_PATH = batch5.LEDGER_PATH
CATALOG_PATH = batch5.CATALOG_PATH
RESEARCH_ROOT = batch5.RESEARCH_ROOT
DEFAULT_QUOTES = os.path.join(RESEARCH_ROOT, DEFAULT_RUN_ID, "quotes.json")

_SIGNAL_SOURCE_PATH = os.path.relpath(__file__, REPO_ROOT).replace(os.sep, "/")

_NYSE_CALENDAR = mcal.get_calendar("NYSE")


def _today_et(now: pd.Timestamp | None = None) -> str:
    """Today's date in New York (YYYY-MM-DD). A module-level seam so tests can
    pin the clock; `now` (tz-aware) injects an instant. compute_and_record reads
    it exactly ONCE per run so every date rule sees the same day."""
    moment = pd.Timestamp.now(tz="UTC") if now is None else now
    return moment.tz_convert("America/New_York").strftime("%Y-%m-%d")


def _most_recent_available_session(before_date: str) -> str:
    """The most recent real NYSE session STRICTLY BEFORE `before_date`.
    Mirrors the sibling intraday drivers' identically-named local helper
    (each driver keeps its own local copy)."""
    before = pd.Timestamp(before_date).date()
    schedule = _NYSE_CALENDAR.schedule(
        start_date=(pd.Timestamp(before_date) - pd.Timedelta(days=14)).date(),
        end_date=before_date)
    sessions = [ts.date() for ts in schedule.index if ts.date() < before]
    if not sessions:
        raise ValueError(
            f"_most_recent_available_session: no real NYSE session found "
            f"in the 14 days before {before_date}")
    return sessions[-1].isoformat()


def fetch_full_history_bars(*, today: str, start_date: str | None = None,
                            end_date: str | None = None) -> tuple:
    """Fetch SPY's real 1-minute RTH bars from `start_date` (default
    `REAL_DATA_START`) through `end_date` (default: the most recent real
    session strictly before `today`). Both overrides are dry-run-only
    (`compute_and_record` refuses them otherwise). Returns
    `(bars, resolved_end_date)`."""
    resolved_end = end_date or _most_recent_available_session(today)
    bars = intraday_data.fetch_intraday_bars(
        TRADED_TICKER, start_date or REAL_DATA_START, resolved_end)
    return bars, resolved_end


def run_real_intraday_backtest(bars: pd.DataFrame, *, cost_bps: float) -> pd.DataFrame:
    """Run the EXISTING, UNMODIFIED RSI(2) signal factory -- every parameter
    passed explicitly from `FACTORY_KWARGS` -- through
    `intraday_backtest_engine.run_intraday_backtest`. A FRESH `signal_fn` per
    call: its closure carries per-session position state that must start
    clean.

    `tax_haircut_rate` is ALWAYS 0.0 and is not a parameter: ABI-1's own tax
    model applies tax exactly once from PRE-tax `net_ret`, so an engine-level
    haircut would double-tax every winner.
    Returns only `trades_df`; the daily curve is built from it by
    `mining.day_trading_abi1_bridge.build_day_trading_curve`."""
    signal_fn = mod.make_intraday_rsi2_reversion_daytrade_signal_fn(**FACTORY_KWARGS)
    trades_df, _engine_daily_returns = run_intraday_backtest(
        bars, signal_fn, cost_bps=cost_bps, tax_haircut_rate=0.0,
        max_hold_days=PARAMETERS["max_hold_days"], direction=DIRECTION)
    return trades_df


def count_force_flat_bar_entries(trades_df: pd.DataFrame) -> int:
    """Trades that FILLED on or after the force-flat bar. An oversold reading
    at the 15:54 bucket end passes the factory's entry guard (< force_flat_et),
    fills at the 15:55 open and is force-flattened on that same bar: a
    one-minute round trip that still pays the full round-trip cost. The
    mechanism is silent on it, so it is counted, not hidden."""
    if trades_df.empty:
        return 0
    filled = trades_df["entry_datetime"].dt.strftime("%H:%M")
    return int((filled >= PARAMETERS["force_flat_et"]).sum())


def incomplete_sessions(bars: pd.DataFrame) -> list:
    """ISO dates of sessions whose minute timestamps are NOT exactly the NYSE
    calendar's scheduled one-minute grid (390 bars on a full day, 210 on a 13:00
    early close): a missing minute, a duplicate, an extra minute, a session cut
    short, or a date absent from the calendar. A row COUNT is not enough -- one
    missing minute plus one duplicated timestamp still has the scheduled count.
    The signal factory finds 5-minute bucket ends by COUNTING observed rows from
    the session's first bar, so after a missing minute every later boundary in
    that session runs a minute late: causal (no look-ahead), but not the
    5-minute bars the mechanism describes. Counted so a real run discloses it."""
    if bars.empty:
        return []
    index = bars.index
    if index.tz is None:
        index = index.tz_localize("America/New_York")
    et_dates = index.tz_convert("America/New_York").date
    minutes_utc = pd.Series(index.tz_convert("UTC").asi8).groupby(et_dates)
    days = sorted(minutes_utc.groups)
    schedule = _NYSE_CALENDAR.schedule(start_date=days[0], end_date=days[-1])
    scheduled = {ts.date(): (row["market_open"], row["market_close"])
                 for ts, row in schedule.iterrows()}
    flagged = []
    for day, minutes in minutes_utc:
        window = scheduled.get(day)
        if window is None:                       # a date the calendar has no session for
            flagged.append(day)
            continue
        expected = pd.date_range(window[0], window[1], freq="1min", inclusive="left")
        got = pd.DatetimeIndex(sorted(minutes.to_numpy()), tz="UTC")
        if len(got) != len(expected) or not (got == expected).all():
            flagged.append(day)
    return [day.isoformat() for day in sorted(flagged)]


def count_session_open_entries(trades_df: pd.DataFrame) -> int:
    """Trades whose entry filled at the 09:30 session open. The mechanism
    forbids any entry before `earliest_entry_et` (10:00), so such a fill can
    only be a signal from the PREVIOUS session's last bar -- the factory's own
    disclosed residual on early-close half days (last bar well before the
    15:55 force-flat; the engine then fills the next session's open). Counted
    so a real run discloses how often it happened rather than hiding it."""
    if trades_df.empty:
        return 0
    return int((trades_df["entry_datetime"].dt.strftime("%H:%M") == "09:30").sum())


_NON_REGULAR_MARKERS = ("pre-market", "premarket", "pre market", "after-hours",
                        "after hours", "post-market", "extended")


def _refuse_unmeasured_quotes(quotes_payload: dict, tested_date: str) -> None:
    """A real run prices a permanent, one-shot verdict off ONE live quote
    sample. `refuse_synthetic_quotes` only catches the literal SYNTHETIC
    marker; a closed-market or old capture would otherwise pass. Require an
    open-market state and a sample taken on the tested date."""
    state = str(quotes_payload.get("market_state", "")).strip().lower()
    if not state.startswith("open") or any(m in state for m in _NON_REGULAR_MARKERS):
        raise ValueError(
            f"refusing a real run: quotes market_state is "
            f"{quotes_payload.get('market_state')!r}, not an open regular "
            "session -- a recorded trial must be priced from a live sample")
    sampled = str(quotes_payload.get("sampled_at_local", ""))[:10]
    if sampled != tested_date:
        raise ValueError(
            f"refusing a real run: quotes sampled_at_local is "
            f"{quotes_payload.get('sampled_at_local')!r} but tested_date is "
            f"{tested_date!r} -- capture the quote on the day of the run")


def _registration_drift(entry: dict) -> str | None:
    """None when this driver still matches the catalog entry it was
    registered against AND pins every parameter the signal factory accepts;
    otherwise the reason. `batch5.parameter_drift` compares only the
    `parameters` dict, so the catalog's `tickers` and `account_types` (the
    tax treatment) are checked too."""
    drift = batch5.parameter_drift(PARAMETERS, entry.get("parameters"))
    if drift is not None:
        return drift
    catalog_tickers = sorted(entry.get("tickers") or [])
    if catalog_tickers != sorted(TRADED):
        return (f"driver traded tickers {sorted(TRADED)} != catalog tickers "
                f"{catalog_tickers}")
    if sorted(entry.get("account_types") or []) != sorted(ACCOUNT_TYPES):
        return (f"driver account_types {sorted(ACCOUNT_TYPES)} != catalog "
                f"account_types {sorted(entry.get('account_types') or [])}")
    accepted = set(inspect.signature(mod.make_intraday_rsi2_reversion_daytrade_signal_fn).parameters)
    unpinned = sorted(accepted ^ set(FACTORY_KWARGS))
    if unpinned:
        return (f"signal factory parameters {unpinned} are not pinned by this "
                "driver's FACTORY_KWARGS -- an unpinned knob would be scored "
                "at whatever default the factory carries")
    return None


def compute_and_record(*, catalog_path: str, quotes_path: str, ledger_path: str,
                       run_id: str, tested_date: str | None, dry_run: bool,
                       end_date: str | None, signal_source_path: str,
                       signal_source_sha256: str,
                       research_root: str = RESEARCH_ROOT,
                       start_date: str | None = None,
                       accept_long_side_only: bool = False) -> dict:
    today = _today_et()          # the ONE clock read of this run
    attempt_id = uuid.uuid4().hex[:16]   # unique per invocation; persisted in the ledger note
    if not dry_run and not tested_date:
        raise ValueError(
            "compute_and_record: a real run needs an explicit tested_date -- "
            "it is written into the permanent ledger row, so it is never "
            "defaulted")
    if not dry_run and tested_date != today:
        raise ValueError(
            f"compute_and_record: a real run is dated today (ET, {today}), "
            f"but tested_date is {tested_date!r} -- that date is not today. "
            "Matching a quote's date to an old tested_date would price the "
            "permanent verdict off stale costs")
    if not dry_run and not accept_long_side_only:
        raise ValueError(
            "compute_and_record: a real run spends this family's ONLY ABI-1 "
            "scoring attempt on the long side alone (the catalog mechanism's "
            "mirrored short side is not implemented by the signal factory, "
            "and a short side would need its own family_id). Pass "
            "accept_long_side_only=True (--accept-long-side-only) to confirm "
            "that is intended")
    if start_date is not None and not dry_run:
        raise ValueError(
            "compute_and_record: start_date override is dry-run-only -- a "
            "real (non-dry-run) scoring attempt must always use the "
            "measured REAL_DATA_START floor, never a shortened window")
    if end_date is not None and not dry_run:
        raise ValueError(
            "compute_and_record: end_date override is dry-run-only -- a "
            "real (non-dry-run) scoring attempt must always run through the "
            "most recent real session, never a truncated window that would "
            "become this family's permanent one-shot outcome")
    if end_date is not None and end_date >= today:
        raise ValueError(
            f"compute_and_record: end_date {end_date!r} is today ({today}) or "
            "later -- fetching a still-forming session would write a PARTIAL "
            "day into the shared minute cache, which is then trusted forever")

    family_id = FAMILY_ID
    catalog_entries = {e["family_id"]: e for e in catalog_mod.load_catalog(catalog_path)}
    entry = catalog_entries.get(family_id)
    if entry is None:
        return {"family_id": family_id, "reason_code": "not_in_catalog",
                "reason": f"{family_id} is not in the catalog"}
    if entry.get("status") != "eligible":
        return {"family_id": family_id, "reason_code": "catalog_status",
                "reason": f"catalog status is {entry.get('status')!r}, not eligible"}
    drift = _registration_drift(entry)
    if drift is not None:
        return {"family_id": family_id, "reason_code": "parameter_drift", "reason": drift}

    already_scored = {
        row["family_id"] for row in trial_ledger.read_all(ledger_path)
        if row.get("event") == "outcome" and row.get("construction") == "ABI-1"}
    if family_id in already_scored:
        return {"family_id": family_id, "reason_code": "already_scored_abi1",
                "reason": "a prior run already recorded a construction=\"ABI-1\" outcome"}

    quotes_payload = batch5.load_quotes(quotes_path)
    batch5.refuse_synthetic_quotes(quotes_payload, dry_run=dry_run)
    if not dry_run:
        _refuse_unmeasured_quotes(quotes_payload, tested_date)
    costs = batch5.execution_costs_for([TRADED_TICKER], quotes_payload)
    cost_bps = costs[TRADED_TICKER]["cost_bps"]
    benchmark_entry_cost_bps = costs[MARKET_BENCHMARK]["spread_bps"] / 2.0

    timing = {}
    started = time.perf_counter()
    bars, resolved_end = fetch_full_history_bars(
        today=today, start_date=start_date, end_date=end_date)
    # yfinance's newest daily row is an in-progress bar when fetched during
    # the session, and load_frame caches it. A real run must never score
    # against a cache left by another day (or by a dry run), so it gets its
    # own per-tested-date directory; a dry run keeps the shared one.
    cache_dir = os.path.join(research_root, run_id, "artifacts",
                             "ohlc" if dry_run else f"ohlc-{tested_date}")
    spy_daily_frame, _spy_source = batch5.load_frame(
        MARKET_BENCHMARK, ohlc_loader=batch5.default_ohlc_loader, cache_dir=cache_dir)
    timing["fetch"] = time.perf_counter() - started

    # The minute cache trusts any existing day file forever, so a session cached
    # while still forming is a PARTIAL day. The newest session is the one most
    # exposed to that and it is scored into a permanent row: a real run refuses
    # when its last session is short of the scheduled minutes (a dry run only
    # reports it). A genuine one-minute gap on the last day also refuses --
    # retry once the next session is available.
    gappy_sessions = incomplete_sessions(bars)
    last_session_iso = max(bars.index.date).isoformat()
    if not dry_run and gappy_sessions and gappy_sessions[-1] == last_session_iso:
        raise ValueError(
            f"refusing a real run: the last intraday session {last_session_iso} is "
            "incomplete (fewer bars than the NYSE schedule) -- likely a partial day "
            "cached while it was still forming. Delete that day's cache file "
            f"under data/mining/intraday_cache/{TRADED_TICKER}/ and re-fetch, or "
            "retry after the next session")

    # batch5.load_frame reuses ANY cached SPY.csv it finds under this run-id
    # (a dry run leaves one). A benchmark that ends before the last intraday
    # session would silently drop the newest sessions from a permanent score.
    # A row dated today is an in-progress bar: never score it, whatever the cache.
    spy_daily_frame = spy_daily_frame[spy_daily_frame.index < pd.Timestamp(today)]
    last_session = pd.Timestamp(max(bars.index.date))
    if spy_daily_frame.empty or spy_daily_frame.index.max() < last_session:
        cached = os.path.join(cache_dir, batch5._cache_name(MARKET_BENCHMARK))
        if os.path.exists(cached):
            os.remove(cached)       # so a retry re-fetches instead of refusing again
        ends = "is empty" if spy_daily_frame.empty else f"ends {spy_daily_frame.index.max().date()}"
        raise ValueError(
            f"{MARKET_BENCHMARK} benchmark frame {ends}, before the last intraday "
            f"session {last_session.date()} -- a stale cached {MARKET_BENCHMARK}.csv "
            f"under {cache_dir}? it has been deleted; re-run so the benchmark is "
            "re-fetched")

    started = time.perf_counter()
    trades_df = run_real_intraday_backtest(bars, cost_bps=cost_bps)
    timing["backtest"] = time.perf_counter() - started

    started = time.perf_counter()
    calendar_index = pd.DatetimeIndex(sorted(set(bars.index.date)))
    curve = bridge.build_day_trading_curve(trades_df, calendar_index)
    account_type = ACCOUNT_TYPE
    result = abi1_driver.evaluate_family(
        curve, benchmark_frame=spy_daily_frame,
        benchmark_entry_cost_bps=benchmark_entry_cost_bps,
        account_type=account_type, tax_policy=TAX_POLICY)
    timing["bridge_and_evaluate"] = time.perf_counter() - started

    session_open_fills = count_session_open_entries(trades_df)
    force_flat_bar_fills = count_force_flat_bar_entries(trades_df)
    last_bar = bars.index[-1].strftime("%Y-%m-%d %H:%M")
    extra_note = (
        f"cost model: {TRADED_TICKER}={cost_bps:.2f}bps "
        f"({batch5.quotes_cost_source(quotes_payload)}); "
        f"real {TRADED_TICKER} intraday history "
        f"{start_date or REAL_DATA_START}..{resolved_end} "
        "(Alpaca SIP 1-minute RTH-only bars, real market data); daily curve "
        "via mining.day_trading_abi1_bridge.build_day_trading_curve; "
        "long side only (the catalog mechanism's mirrored short side is not "
        "implemented by the signal factory and is NOT tested here); "
        f"details: entries filled at the session open={session_open_fills}; "
        f"entries filled on the force-flat bar={force_flat_bar_fills}; "
        f"sessions with missing minutes={len(gappy_sessions)} (bucket boundaries "
        "after a missing minute run late for the rest of that session); "
        f"last intraday bar={last_bar}; "
        f"attempt={attempt_id}; "
        f"code: driver={signal_source_path}@{signal_source_sha256[:12]}; "
        f"signals=day_trading_signals.py@{evidence.source_sha256(mod.__file__)[:12]}")

    ledger_post_write_error = None
    if dry_run:
        verdict = result["verdict"]
        stage = verdict if verdict in evidence.SCORED_STAGES else "insufficient_evidence"
    else:
        # A concurrent invocation can pass the already_scored prefilter above
        # (unlocked, cheap, run before this expensive compute) and still lose
        # the atomic check-and-append inside record_abi1_outcome --
        # OutcomeAlreadyRecorded means this attempt's freshly computed result
        # must be discarded, same as the early check's own skip.
        try:
            stage = abi1_driver.record_abi1_outcome(
                ledger_path, family_id, run_id, tested_date, result,
                catalog_entry=entry, instrument_set=list(TRADED),
                cash_convention=CASH_CONVENTION, benchmark=MARKET_BENCHMARK,
                tax_policy=TAX_POLICY, ticker_label=None, extra_note=extra_note,
                # The attempt token rides in the source path too: UNSCORED rows persist
                # no note, only evidence.observed.signal_source_path.
                signal_source_path=f"{signal_source_path}#attempt={attempt_id}",
                signal_source_sha256=signal_source_sha256)
        except trial_ledger.OutcomeAlreadyRecorded:
            return {"family_id": family_id, "reason_code": "already_scored_abi1",
                    "reason": "a concurrent run recorded a construction=\"ABI-1\" "
                              "outcome first -- this attempt's freshly computed "
                              "result is discarded"}
        except Exception as exc:
            # record_abi1_outcome appends the permanent OUTCOME row first and
            # registers the first_test row second. Decide from the ledger which
            # case this is: our own outcome exists (the slot is spent -- surface
            # the result and the error instead of losing them with the
            # traceback); only ANOTHER run's outcome exists (a concurrent
            # winner -- discard ours, like OutcomeAlreadyRecorded); none exists
            # (nothing was written -- the failure simply propagates); or the
            # ledger cannot be read (state unknown -- surface everything).
            verdict = result["verdict"]
            stage = verdict if verdict in evidence.SCORED_STAGES else "insufficient_evidence"
            try:
                abi1_rows = [
                    row for row in trial_ledger.read_all(ledger_path)
                    if row.get("event") == "outcome" and row.get("family_id") == family_id
                    and row.get("construction") == "ABI-1"]
            except Exception as read_exc:
                ledger_post_write_error = (
                    f"{type(exc).__name__}: {exc}; ledger state unknown -- "
                    f"re-reading it failed ({type(read_exc).__name__}: {read_exc}); "
                    "verify the ledger before any retry")
            else:
                if not abi1_rows:
                    raise
                if not any(f"attempt={attempt_id}" in json.dumps(row, default=str)
                           for row in abi1_rows):
                    return {"family_id": family_id, "reason_code": "already_scored_abi1",
                            "reason": "this attempt's ledger write raised "
                                      f"({type(exc).__name__}: {exc}) and another run's "
                                      "construction=\"ABI-1\" outcome exists -- this "
                                      "attempt's result is discarded"}
                ledger_post_write_error = f"{type(exc).__name__}: {exc}"

    layer1, holdout, window = result["layer1"], result["holdout"], result["window"]
    account_index = result["account_daily"].index
    return {
        "family_id": family_id, "shape": SHAPE, "tickers": list(TRADED),
        "direction": DIRECTION,
        "verdict": result["verdict"], "stage": stage,
        "dsr_value": layer1.get("dsr_value"),
        "sharpe_daily_excess": layer1.get("sharpe_daily_excess"),
        "n_sessions": layer1.get("n_sessions"),
        "n_trades": int(len(trades_df)),
        "entries_filled_at_session_open": session_open_fills,
        "entries_filled_on_force_flat_bar": force_flat_bar_fills,
        "last_intraday_bar": last_bar,
        "n_incomplete_sessions": len(gappy_sessions),
        "incomplete_sessions": gappy_sessions,
        "account_start": account_index[0].strftime("%Y-%m-%d") if len(account_index) else None,
        "account_end": account_index[-1].strftime("%Y-%m-%d") if len(account_index) else None,
        "time_in_market": result.get("time_in_market"),
        "tax_policy": TAX_POLICY, "cash_convention": CASH_CONVENTION,
        "account_type": account_type,
        "benchmark": MARKET_BENCHMARK, "cost_model_note": extra_note,
        "cost_bps_round_trip": cost_bps,
        "holdout_evaluated": holdout.get("evaluated"),
        "holdout_passed": holdout.get("passed"),
        "holdout_psr": holdout.get("psr"),
        "window": {
            "account_sessions": window.get("account_sessions"),
            "scored_sessions": window.get("scored_sessions"),
            "in_sample_sessions": window.get("in_sample_sessions"),
            "holdout_sessions": window.get("holdout_sessions"),
            "sessions_dropped_to_benchmark": window.get("sessions_dropped_to_benchmark"),
            "benchmark_truncated": window.get("benchmark_truncated"),
        },
        "real_intraday_data_start": start_date or REAL_DATA_START,
        "real_intraday_data_end": resolved_end,
        "n_intraday_sessions": int(len(calendar_index)),
        "timing_seconds": timing,
        "ledger_post_write_error": ledger_post_write_error,
        "attempt_id": attempt_id,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quotes", default=DEFAULT_QUOTES)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--tested-date", default=None,
        help="REQUIRED for a real run (written into the permanent ledger "
             "row; must equal the day the quote sample was taken).")
    parser.add_argument(
        "--accept-long-side-only", action="store_true",
        help="REQUIRED for a real run: confirms the family's one ABI-1 slot "
             "is spent on the long side alone (see the module docstring).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ledger", default=LEDGER_PATH)
    parser.add_argument("--catalog", default=CATALOG_PATH)
    parser.add_argument("--research-root", default=RESEARCH_ROOT)
    parser.add_argument(
        "--end-date", default=None,
        help="DRY-RUN-ONLY override for the intraday fetch's end date "
             "(YYYY-MM-DD). Default is the most recent real NYSE session "
             "strictly before today. Rejected outside --dry-run, and "
             "rejected when today or later (a partial day would poison the "
             "shared minute cache).")
    parser.add_argument(
        "--start-date", default=None,
        help="DRY-RUN-ONLY override for the intraday fetch's start date "
             "(YYYY-MM-DD). Rejected outside --dry-run.")
    args = parser.parse_args(argv)

    result = compute_and_record(
        catalog_path=args.catalog, quotes_path=args.quotes,
        ledger_path=args.ledger, run_id=args.run_id, tested_date=args.tested_date,
        dry_run=args.dry_run, end_date=args.end_date, start_date=args.start_date,
        signal_source_path=_SIGNAL_SOURCE_PATH,
        signal_source_sha256=evidence.source_sha256(__file__),
        research_root=args.research_root,
        accept_long_side_only=args.accept_long_side_only)

    safe_result = batch5._json_safe(result)
    if "reason_code" in result:
        # A skip dict carries no scoring output. Writing it would overwrite a
        # prior real run's results file with a reason-only payload, so it is
        # printed only.
        print(json.dumps({"output": None, "result": safe_result}))
        return 0

    run_dir = os.path.join(args.research_root, args.run_id)
    payload = {
        "schema_version": 1, "construction": "ABI-1", "run_id": args.run_id,
        "tested_date": args.tested_date, "dry_run": args.dry_run,
        "driver": _SIGNAL_SOURCE_PATH, "result": safe_result,
    }
    output_path = os.path.join(
        run_dir,
        "intraday-rsi2-reversion-dry-run.json" if args.dry_run
        else "intraday-rsi2-reversion-results.json")
    try:
        os.makedirs(run_dir, exist_ok=True)
        batch3._atomic_write_text(
            output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    except Exception as exc:
        # On a real run the permanent ledger row is ALREADY written; the full
        # result must survive somewhere, so print it and fail loudly.
        print(json.dumps({"output": None, "write_error": f"{type(exc).__name__}: {exc}",
                          "result": safe_result}))
        return 1
    print(json.dumps({"output": output_path.replace(os.sep, "/"), "result": safe_result}))
    # The outcome row is permanent but its first_test registration failed: the
    # result is printed and written above; a non-zero exit makes it loud.
    return 1 if safe_result.get("ledger_post_write_error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
