"""Opt-in "end the scored price history at a chosen last session".

Why this exists: a data vendor can lack a single daily bar for many series
while having later bars. The scoring drivers refuse a history
with a calendar hole (refused, never trimmed), and that default is unchanged.
This module backs an explicit ``--history-end YYYY-MM-DD`` setting a driver can
be given instead. It only DROPS sessions dated after the cutoff. It adds, moves
or repairs no bar, and every calendar / hole / final-window check a driver runs
keeps running on the trimmed frames.

Rules the functions here enforce:

* Off is ``None`` and nothing else. ``trim_frames_to_history_end(frames, None)``
  hands back the SAME frame objects, so the default path is byte-identical.
* Fail closed. The cutoff must be a strict ``YYYY-MM-DD`` date and an actual NYSE
  session, and EVERY series must hold a bar dated exactly on it; otherwise a
  ``HistoryCutoffError`` carrying a machine-readable ``reason_code`` is raised
  (every offending symbol is collected into the one error).
* Pure and in-memory. No file I/O, inputs are never mutated, and the trimmed
  frames are copies. The caller trims AFTER a frame was loaded and cached, never
  before the cache write, so the cached history stays complete.

A scored history end is PERMANENT for the account it scores: the outcome row's
account ends at the cutoff, and a NEW family's first_test pin (its ABI-1 holdout
is the tail of that account) ends there too, so the family cannot be re-scored on
longer history. The pin is not unconditional: for a family that already has a
first_test row ``trial_ledger.record_first_test`` is a no-op, and a not_scoreable
row pins none. ``history_end_clause`` states the permanence in the text a driver
writes into its ledger note. ``check_real_run_guard`` refuses a cutoff that would
spend that one-shot slot by mistake (a cutoff after the run's tested date, or an
unexpectedly large drop) and ``refusal_payload`` gives every driver one standard
shape for reporting a refused setting. ``validate_guard_arguments`` lets a driver
check the guard's two inputs at its door, before any load.
"""
from __future__ import annotations

import datetime
import re
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .vendor_coverage import index_from_frame

REASON_INVALID = "history_end_invalid"
REASON_NOT_A_SESSION = "history_end_not_a_session"
REASON_NOT_COVERED = "history_end_not_covered"
REASON_EMPTY = "history_end_empty"
REASON_AFTER_TESTED_DATE = "history_end_after_tested_date"
REASON_DROP_TOO_LARGE = "history_end_drop_too_large"
# every reason code this module can raise, so a driver can test membership
HISTORY_END_REASON_CODES = (REASON_INVALID, REASON_NOT_A_SESSION, REASON_NOT_COVERED,
                            REASON_EMPTY, REASON_AFTER_TESTED_DATE,
                            REASON_DROP_TOO_LARGE)

DECISION = "--history-end"
HISTORY_END_MARKER = "HISTORY END (opt-in"
CLAUSE_MAX_CHARS = 700
# a real run refuses a history end that drops more than this many sessions from
# any one series unless the caller passes a larger max_drop explicitly
MAX_UNACCEPTED_DROP_SESSIONS = 10
# the shared rule a not-covered / empty refusal rests on (stated once in an
# aggregated refusal) and how many offenders that refusal lists inline
NOT_COVERED_RULE = ("a history end needs a real bar on that date in every "
                    "series")
INLINE_OFFENDER_LIMIT = 10

_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_SESSION_PAD = pd.Timedelta(days=7)
_RECORD_KEYS = ("raw_first", "raw_last", "kept_last", "dropped_sessions",
                "dropped_dates")


class HistoryCutoffError(ValueError):
    """A history-end setting this code refuses. A ``ValueError`` (every existing
    ``except ValueError`` still catches it) that also carries a machine-readable
    ``reason_code`` (one of ``HISTORY_END_REASON_CODES``: ``history_end_invalid``,
    ``history_end_not_a_session``, ``history_end_not_covered``,
    ``history_end_empty``, ``history_end_after_tested_date`` or
    ``history_end_drop_too_large``) and the offending ``symbols`` (a tuple, empty
    when the refusal is not about a series). ``brief`` is an optional one-line
    form of ``message`` (default: the message itself) that an aggregated refusal
    lists per offender, so a wide pool's error text stays short."""

    def __init__(self, reason_code: str, message: str, *, symbols=(), brief=None):
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.symbols = tuple(symbols)
        self.brief = str(message) if brief is None else str(brief)


def _iso(stamp) -> str:
    return f"{pd.Timestamp(stamp):%Y-%m-%d}"


def _invalid(message: str) -> HistoryCutoffError:
    return HistoryCutoffError(
        REASON_INVALID,
        f"history end must be a real calendar date written YYYY-MM-DD (or unset): "
        f"{message}")


# ------------------------------------------------------------------- parsing

def _calendar_day(value) -> datetime.date:
    """The calendar day a setting value names, or ``HistoryCutoffError``
    (``history_end_invalid``). A string must be strictly YYYY-MM-DD; a date is
    taken as is; a datetime / Timestamp only when it is tz-naive and exactly
    midnight (a time of day or a time zone would need a guess about which date
    was meant)."""
    if isinstance(value, str):
        if not _ISO_DATE.fullmatch(value):
            raise _invalid(f"got {value!r}")
        try:
            return datetime.date(int(value[0:4]), int(value[5:7]), int(value[8:10]))
        except ValueError:
            raise _invalid(f"{value!r} is not a real calendar date") from None
    if isinstance(value, datetime.datetime):          # includes pandas Timestamp
        if value is pd.NaT:
            raise _invalid("got NaT")
        if value.tzinfo is not None:
            raise _invalid(f"got a timezone-aware value {value!r}")
        if (value.hour or value.minute or value.second or value.microsecond
                or getattr(value, "nanosecond", 0)):
            raise _invalid(f"got a value with a time of day {value!r}")
        return value.date()
    if isinstance(value, datetime.date):
        return value
    raise _invalid(f"got {type(value).__name__} {value!r}")


def _require_session(stamp: pd.Timestamp) -> None:
    from mining.batch5.treasury_month_end_index_demand_overlay import exchange_sessions
    try:
        # a one-day window is rejected by exchange_calendars (start must precede
        # end), so look at a week either side and test membership
        is_session = stamp in exchange_sessions(stamp - _SESSION_PAD,
                                                stamp + _SESSION_PAD)
    except ImportError:
        raise
    except Exception as exc:                # calendar cannot cover the date
        raise HistoryCutoffError(
            REASON_NOT_A_SESSION,
            f"history end {_iso(stamp)} could not be confirmed as an NYSE session "
            f"({type(exc).__name__})") from None
    if not is_session:
        raise HistoryCutoffError(
            REASON_NOT_A_SESSION,
            f"history end {_iso(stamp)} is not an NYSE session (a weekend, an "
            "exchange holiday or an unscheduled closure)")


def parse_history_end(value):
    """The cutoff as a normalized, tz-naive ``Timestamp``, or ``None`` for OFF.

    ``None`` is the only off value: an empty string, a wrong format, a time of
    day, a time zone or any other type is ``history_end_invalid``. A valid date
    that is not an actual NYSE session (weekend, holiday, unscheduled closure,
    or outside what a timestamp can represent) is ``history_end_not_a_session``.
    """
    if value is None:
        return None
    day = _calendar_day(value)
    try:
        stamp = pd.Timestamp(day)
    except ValueError:                       # a real date no Timestamp can hold
        raise HistoryCutoffError(
            REASON_NOT_A_SESSION,
            f"history end {day.isoformat()} cannot be an NYSE session") from None
    _require_session(stamp)
    return stamp


# ------------------------------------------------------------------ trimming

def _checked_cutoff(cutoff_ts) -> pd.Timestamp:
    """``cutoff_ts`` must be what ``parse_history_end`` returns: a tz-naive
    midnight Timestamp. Anything else is refused rather than guessed at."""
    if (not isinstance(cutoff_ts, pd.Timestamp) or cutoff_ts is pd.NaT
            or cutoff_ts.tzinfo is not None or cutoff_ts != cutoff_ts.normalize()):
        raise HistoryCutoffError(
            REASON_INVALID,
            "an already-parsed cutoff (the tz-naive midnight Timestamp that "
            f"parse_history_end returns) was expected, got {cutoff_ts!r}")
    return cutoff_ts


def trim_frame_to_history_end(frame, cutoff_ts, *, symbol: str = ""):
    """``(trimmed_copy, record)`` for one OHLC frame.

    Keeps every row dated on or before ``cutoff_ts`` and drops the rows after
    it; the kept rows are exactly the original ones (values, dtypes, index).
    Bar dates are read the way the drivers read them (``index_from_frame``: a
    ``Date`` column wins, else a DatetimeIndex). The frame must hold a bar dated
    exactly on the cutoff (``history_end_not_covered``); a frame with no rows is
    ``history_end_empty``. ``record`` holds ISO-string values: ``raw_first``,
    ``raw_last``, ``kept_last``, ``dropped_sessions`` (the number of distinct
    dates dropped) and ``dropped_dates`` (sorted, may be empty)."""
    cutoff = _checked_cutoff(cutoff_ts)
    label = symbol or "series"
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{label}: expected an OHLC DataFrame")
    if len(frame) == 0:
        raise HistoryCutoffError(
            REASON_EMPTY, f"{label}: the frame has no rows to end at "
            f"{_iso(cutoff)}", symbols=(symbol,),
            brief=f"{label}: the frame has no rows")
    dates = index_from_frame(frame)
    if dates.isna().any():
        raise ValueError(f"{label}: bar dates are unreadable (NaT); refusing "
                         "to guess which rows fall after the cutoff")
    if not (dates == cutoff).any():
        runs = f"{_iso(dates.min())}..{_iso(dates.max())}"
        raise HistoryCutoffError(
            REASON_NOT_COVERED,
            f"{label}: no bar dated exactly {_iso(cutoff)} (its history runs "
            f"{runs}); {NOT_COVERED_RULE}", symbols=(symbol,),
            brief=f"{label}: no bar dated exactly {_iso(cutoff)} "
                  f"(history runs {runs})")
    keep = np.asarray(dates <= cutoff)
    dropped = sorted({_iso(d) for d in dates[~keep]})
    record = {
        "raw_first": _iso(dates.min()),
        "raw_last": _iso(dates.max()),
        "kept_last": _iso(dates[keep].max()),
        "dropped_sessions": len(dropped),
        "dropped_dates": dropped,
    }
    return frame.loc[keep].copy(), record


def trim_frames_to_history_end(frames, history_end):
    """``(frames_out, provenance_or_None)`` for a mapping of symbol -> frame.

    ``history_end`` None is OFF: a new dict holding the SAME frame objects and no
    provenance, so the default path is unchanged. Otherwise the cutoff is parsed
    (``parse_history_end``), every frame is trimmed, and ALL failures across
    symbols are collected into ONE ``HistoryCutoffError`` naming every offending
    symbol (``reason_code`` is ``history_end_not_covered`` unless every offender
    is ``history_end_empty``). An empty mapping is refused too: a cutoff no
    series covers is not "covered".

    The error's ``symbols`` tuple always holds EVERY offender, sorted, but its
    text is bounded however wide the pool is: the shared rule is stated once,
    at most ``INLINE_OFFENDER_LIMIT`` offenders (the first ones in sorted order)
    are listed inline, each with its own short reason, and the rest are counted
    ("and N more (see symbols)")."""
    cutoff = parse_history_end(history_end)
    if cutoff is None:
        return dict(frames), None
    if not isinstance(frames, Mapping) or not frames:
        raise HistoryCutoffError(
            REASON_NOT_COVERED,
            f"history end {_iso(cutoff)} was given but no series were supplied "
            "to end there")
    trimmed, records, failures = {}, {}, {}
    for symbol, frame in frames.items():
        try:
            trimmed[symbol], records[str(symbol)] = trim_frame_to_history_end(
                frame, cutoff, symbol=str(symbol))
        except HistoryCutoffError as exc:
            failures[str(symbol)] = exc
    if failures:
        codes = {exc.reason_code for exc in failures.values()}
        code = codes.pop() if len(codes) == 1 else REASON_NOT_COVERED
        offenders = tuple(sorted(failures))
        # each brief already starts with its own symbol
        listed = [failures[s].brief for s in offenders[:INLINE_OFFENDER_LIMIT]]
        if len(offenders) > INLINE_OFFENDER_LIMIT:
            listed.append(f"and {len(offenders) - INLINE_OFFENDER_LIMIT} more "
                          "(see symbols)")
        raise HistoryCutoffError(
            code, f"history end {_iso(cutoff)} refused for {len(offenders)} "
            f"series ({NOT_COVERED_RULE}): {'; '.join(listed)}", symbols=offenders)
    return trimmed, build_provenance(cutoff, records)


# ---------------------------------------------------------------- provenance

def build_provenance(cutoff_ts, records_by_symbol) -> dict:
    """The JSON-serializable statement of what a history end did.

    ``{"history_end", "opt_in": True, "decision", "by_symbol": {sym: record},
    "dropped_dates" (sorted union), "total_dropped_series_sessions"}``. The
    per-symbol records are copied; symbols are listed in sorted order."""
    cutoff = _checked_cutoff(cutoff_ts)
    if not records_by_symbol:
        raise ValueError("build_provenance needs at least one series record")
    by_symbol = {}
    for symbol in sorted(records_by_symbol, key=str):
        source = records_by_symbol[symbol]
        by_symbol[str(symbol)] = {
            "raw_first": str(source["raw_first"]),
            "raw_last": str(source["raw_last"]),
            "kept_last": str(source["kept_last"]),
            "dropped_sessions": int(source["dropped_sessions"]),
            "dropped_dates": sorted(str(d) for d in source["dropped_dates"]),
        }
    return {
        "history_end": _iso(cutoff),
        "opt_in": True,
        "decision": DECISION,
        "by_symbol": by_symbol,
        "dropped_dates": sorted({d for r in by_symbol.values()
                                 for d in r["dropped_dates"]}),
        "total_dropped_series_sessions": sum(
            r["dropped_sessions"] for r in by_symbol.values()),
    }


# -------------------------------------------------------------------- clause

def _render_groups(groups: Mapping, collapse=None) -> str:
    parts = []
    for key, symbols in groups.items():
        if key == collapse:
            parts.append(f"{key}: {len(symbols)} series")
        else:
            parts.append(f"{key}: {', '.join(symbols)}")
    return "; ".join(parts)


def _tiered_listing(pairs, budget: int, *, key_order=str) -> str:
    """A per-series listing of ``(symbol, key)`` pairs (``key`` is text: a date
    or a count) within ``budget`` characters. Tiers, first that fits:
    ``SYM key`` sorted by symbol; grouped by key (lossless, far shorter when a
    pooled run's series share a value), groups ordered by ``key_order``; grouped
    with the largest group reduced to its series count (the outliers stay named,
    which is what a reader needs); finally the grouped text cut to fit and marked
    ``(truncated)``. Full detail is always in the provenance ``by_symbol``."""
    ordered = sorted(pairs)
    plain = ", ".join(f"{symbol} {key}" for symbol, key in ordered)
    if len(plain) <= budget:
        return plain
    groups = {}
    for symbol, key in ordered:
        groups.setdefault(key, []).append(symbol)
    groups = dict(sorted(groups.items(), key=lambda item: key_order(item[0])))
    grouped = _render_groups(groups)
    if len(grouped) <= budget:
        return grouped
    modal = max(groups, key=lambda key: (len(groups[key]), key_order(key)))
    collapsed = _render_groups(groups, collapse=modal)
    if len(collapsed) <= budget:
        return collapsed
    suffix = " (truncated)"
    cut = collapsed[:max(budget - len(suffix), 0)]
    boundary = max(cut.rfind(", "), cut.rfind("; "))
    return (cut[:boundary] if boundary > 0 else cut) + suffix


def _last_bar_listing(by_symbol: Mapping, budget: int) -> str:
    """Each series' raw last bar (an ISO date, so text order is date order),
    within ``budget`` characters -- see ``_tiered_listing`` for the tiers."""
    return _tiered_listing(
        [(symbol, rec["raw_last"]) for symbol, rec in by_symbol.items()], budget)


def _dropped_count_listing(by_symbol: Mapping, budget: int) -> str:
    """Each series' number of dropped sessions, within ``budget`` characters --
    see ``_tiered_listing`` for the tiers. Groups run in numeric count order."""
    return _tiered_listing(
        [(symbol, str(int(rec["dropped_sessions"])))
         for symbol, rec in by_symbol.items()], budget, key_order=int)


def history_end_clause(provenance: Mapping) -> str:
    """One deterministic sentence-group for a ledger note (no dollar figures, at
    most ``CLAUSE_MAX_CHARS`` characters however many series there are). It
    starts with ``HISTORY_END_MARKER`` so a test or reader can find it, and
    states: the history was ended at the cutoff by an explicit opt-in setting
    (the setting's identifier) and whether the vendor history extended past it; the
    later dates dropped (first..last and the count); when the series lost
    DIFFERENT numbers of sessions, how many each lost (so the note shows which
    series lacked which sessions); each series' raw last bar; that the setting
    added, moved or repaired no bar; and that the scored account therefore
    permanently ends at the cutoff (any ABI-1 holdout is its tail; a
    not-scoreable row pins none), so the family cannot be re-scored on longer
    history. The two per-series listings share the length cap: the drop counts
    take at most half of what the fixed text leaves, the raw last bars the rest."""
    cutoff = provenance["history_end"]
    dropped = provenance["dropped_dates"]
    by_symbol = provenance["by_symbol"]
    vendor = ("the vendor history extended past it." if dropped
              else "the vendor history ended there.")
    head = (f"{HISTORY_END_MARKER} setting, {provenance['decision']}): the scored "
            f"history was ended at {cutoff} by an explicit setting; {vendor}")
    if dropped:
        span = dropped[0] if len(dropped) == 1 else f"{dropped[0]}..{dropped[-1]}"
        noun = "date" if len(dropped) == 1 else "dates"
        middle = f" Later dates dropped: {span} ({len(dropped)} distinct {noun})."
    else:
        middle = " Later dates dropped: none."
    tail = (" The setting added, moved or repaired no bar. The scored account "
            f"therefore permanently ends at {cutoff} (any ABI-1 holdout is its "
            "tail): the family cannot be re-scored on longer history.")
    label = " Raw last bars: "
    room = CLAUSE_MAX_CHARS - len(head) - len(middle) - len(tail) - len(label) - 1
    drop_part = ""
    if len({int(rec["dropped_sessions"]) for rec in by_symbol.values()}) > 1:
        drop_label = " Dropped sessions per series: "
        drop_room = room - len(drop_label) - 1
        drop_part = (f"{drop_label}"
                     f"{_dropped_count_listing(by_symbol, drop_room // 2)}.")
        room -= len(drop_part)
    listing = _last_bar_listing(by_symbol, room)
    return f"{head}{middle}{drop_part}{label}{listing}.{tail}"


# ---------------------------------------------------------------- real-run guard

def _tested_day(value) -> datetime.date:
    try:
        return _calendar_day(value)
    except HistoryCutoffError:
        raise HistoryCutoffError(
            REASON_INVALID,
            "tested_date must be a real calendar date written YYYY-MM-DD "
            f"(got {value!r}); the history end cannot be compared with it") from None


def validate_guard_arguments(*, tested_date, max_drop) -> None:
    """Refuse a malformed ``tested_date`` or ``max_drop`` (both required, keyword
    only; ``history_end_invalid``), or return ``None``. ``max_drop`` must be an
    int >= 0 and not a bool; ``tested_date`` a real YYYY-MM-DD string or a date.

    Drivers call this at their door, BEFORE any load, whenever a history end is
    set, so these two inputs can never fail AFTER the load step has written the
    per-run OHLC cache. ``check_real_run_guard`` calls it too, so the two share
    one set of rules and one message text. It has no provenance to look at and
    always validates."""
    if isinstance(max_drop, bool) or not isinstance(max_drop, int) or max_drop < 0:
        raise HistoryCutoffError(
            REASON_INVALID,
            f"max_drop must be a whole number of sessions >= 0, got {max_drop!r}")
    _tested_day(tested_date)


def check_real_run_guard(provenance, *, tested_date,
                         max_drop=MAX_UNACCEPTED_DROP_SESSIONS):
    """Refuse a history end that would permanently spend a one-shot scoring slot
    by mistake. Drivers call this on REAL runs only, with the provenance that
    ``trim_frames_to_history_end`` returned. Returns ``None`` when fine.

    ``provenance`` None (the setting is off) is a no-op that checks nothing, not
    even the other arguments, so the default path is unchanged. Otherwise the
    arguments are checked first by ``validate_guard_arguments`` (``max_drop``
    must be an int >= 0 and not a bool, ``tested_date`` a real YYYY-MM-DD date;
    either failure is ``history_end_invalid``), then, in this order:

    * the cutoff must not be LATER than ``tested_date``
      (``history_end_after_tested_date``; equal is fine);
    * no series may have lost more than ``max_drop`` sessions
      (``history_end_drop_too_large``). The message names the worst series (ties
      go to the first symbol) and its drop and says how to accept a larger one;
      ``symbols`` lists every series over the limit, sorted."""
    if provenance is None:
        return None
    validate_guard_arguments(tested_date=tested_date, max_drop=max_drop)
    tested = _tested_day(tested_date)
    cutoff = datetime.date.fromisoformat(provenance["history_end"])
    if cutoff > tested:
        raise HistoryCutoffError(
            REASON_AFTER_TESTED_DATE,
            f"history end {cutoff.isoformat()} is later than the run's tested date "
            f"{tested.isoformat()}; a scored history cannot end after the day it "
            "is tested (a real run permanently spends its one-shot slot)")
    over = {symbol: int(rec["dropped_sessions"])
            for symbol, rec in provenance["by_symbol"].items()
            if int(rec["dropped_sessions"]) > max_drop}
    if over:
        worst = min(over, key=lambda symbol: (-over[symbol], symbol))
        raise HistoryCutoffError(
            REASON_DROP_TOO_LARGE,
            f"history end {cutoff.isoformat()} would drop {over[worst]} sessions "
            f"from {worst}, more than the {max_drop} a real run accepts without "
            f"being told to ({len(over)} series over the limit); if that drop is "
            "intended, pass --max-history-drop N to accept a larger drop",
            symbols=sorted(over))
    return None


# -------------------------------------------------------------- refusal payload

_CACHE_KEPT_SENTENCE = (
    "The per-run OHLC cache written before this check is kept (no ledger row, "
    "result file or report is written); retry with a fresh --run-id: a reused "
    "run-id reads that cached history instead of fetching again (real runs of "
    "the bespoke overlay drivers refuse it as stale_ohlc_cache).")
# raised AFTER the load step wrote the per-run cache (the two guard refusals need
# the provenance the trim produces); the door refusals happen before any load
_CACHE_KEPT_CODES = (REASON_NOT_COVERED, REASON_EMPTY,
                     REASON_AFTER_TESTED_DATE, REASON_DROP_TOO_LARGE)


def refusal_payload(exc, *, history_end_requested, **extra) -> dict:
    """The one standard shape a driver reports a refused history end in:
    ``{"reason_code", "reason", "history_end_requested", "symbols", **extra}``.

    ``history_end_requested`` is what the caller asked for (the string, or None
    when the setting was off). ``symbols`` is a new list. ``reason`` is the
    error text; for the four refusals raised after the per-run OHLC cache is
    written (not covered, empty, after the tested date, drop too large) it also
    says that the cache is kept and that a retry needs a fresh run id. Any
    ``extra`` fields are merged after the standard ones."""
    if not isinstance(exc, HistoryCutoffError):
        raise TypeError("refusal_payload needs the HistoryCutoffError that was "
                        f"raised, got {type(exc).__name__}")
    reason = str(exc)
    if exc.reason_code in _CACHE_KEPT_CODES:
        reason = reason.rstrip()
        if reason and reason[-1] not in ".!?":
            reason += "."
        reason = f"{reason} {_CACHE_KEPT_SENTENCE}"
    requested = None if history_end_requested is None else str(history_end_requested)
    return {"reason_code": exc.reason_code, "reason": reason,
            "history_end_requested": requested, "symbols": list(exc.symbols),
            **extra}
