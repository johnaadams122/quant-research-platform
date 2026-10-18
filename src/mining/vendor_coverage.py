"""Pre-scoring vendor-history coverage check.

Why this exists: the scoring drivers REFUSE a history whose calendar has a hole
(``calendar_mismatch`` / ``unequal_trailing_coverage`` / ``missing_exchange_session``),
which is correct, but they only say so at the moment a real run starts. A
vendor can lack one interior session for a handful of ETFs and indices while SPY
has it, which blocks every family that reads them. This module answers the
question BEFORE a scoring run: per symbol, which actual
NYSE sessions inside a window are absent, which bars sit on a closed day, and is
the last bar an in-progress (live, partial) bar of today.

It only REPORTS. It never trims, fills or repairs anything. The reference
calendar is ``mining.batch5.treasury_month_end_index_demand_overlay
.exchange_sessions`` (the sessions the exchange actually held, unscheduled
closures omitted) so this check and the drivers agree on what a hole is.

Recipe for a family: pass every series the driver loads -- traded legs, signal
series, the benchmark (SPY) and any financing series (e.g. ^IRX). The treasury
overlay driver, for instance, loads IEF, TLT, AGG, SPY and ^IRX.
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd

MARKET_CLOSE_HOUR_ET = 16        # a bar dated D is final only from D 16:00 ET
_ET = "America/New_York"


def _normalized_index(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def index_from_frame(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Bar dates of an OHLC frame, read the way the scoring drivers read them
    (``run_batch3_mining._normalize_ohlc``): a ``Date`` column wins --
    ``data_pipeline.fetch_ohlc`` returns python dates in a ``Date`` COLUMN over
    a plain integer index -- otherwise the index must already be datetime-like.
    An integer index with no ``Date`` column is refused (it would read as 1970
    epoch dates and report every session as missing)."""
    if "Date" in frame.columns:
        return _normalized_index(pd.to_datetime(frame["Date"]))
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("OHLC frame has neither a Date column nor a DatetimeIndex")
    return _normalized_index(frame.index)


OHLC_COLUMNS = ("Open", "High", "Low", "Close")


def drop_all_blank_rows(frame):
    """``(frame, [dropped ISO dates])``: ``frame`` without the rows whose every
    OHLC field is blank, exactly as the scoring drivers drop them.

    A faithful mirror of ``scripts/run_batch5_mining._drop_all_nan_rows`` (the
    step ``load_frame`` runs on every fresh fetch, before any date is read),
    kept in step with it by a parity test. A row with NO prices at all is a
    session the vendor listed but never printed, and the drivers discard it; a
    row with SOME prices is a bad print, which the drivers keep (and refuse
    later), so it is kept here too. Anything that is not a non-empty DataFrame,
    and a frame with none of the OHLC columns, comes back unchanged. The input
    is never modified."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame, []
    present = [column for column in OHLC_COLUMNS if column in frame.columns]
    if not present:
        return frame, []
    blank = frame[present].isna().all(axis=1)
    if not blank.any():
        return frame, []
    if "Date" in frame.columns:
        stamps = pd.to_datetime(frame.loc[blank, "Date"])
    else:
        stamps = pd.to_datetime(pd.Index(frame.index[blank]))
    dropped = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in stamps]
    return frame.loc[~blank], dropped


def driver_index_from_frame(frame) -> pd.DatetimeIndex:
    """Bar dates the way the scoring drivers COUNT a bar: rows with no prices at
    all are dropped first (``drop_all_blank_rows``), then the dates are read as
    ``index_from_frame`` reads them. This is what the coverage preflight uses, so
    it never calls a session present that a driver would then refuse. A frame
    with no row holding any price is a ``ValueError`` (a driver refuses it as
    empty). ``index_from_frame`` itself is unchanged: it still reads EVERY row,
    because ``history_cutoff`` builds a row mask from it and needs one date per
    row of the frame it trims."""
    kept, _dropped = drop_all_blank_rows(frame)
    if isinstance(kept, pd.DataFrame) and len(kept) == 0:
        raise ValueError("OHLC frame has no row with any price")
    return index_from_frame(kept)


def _as_of_et(as_of) -> pd.Timestamp:
    stamp = pd.Timestamp(as_of)
    if stamp.tzinfo is None:
        return stamp.tz_localize(_ET)
    return stamp.tz_convert(_ET)


def _iso(dates) -> list:
    return [f"{d:%Y-%m-%d}" for d in dates]


def coverage_report(indices: Mapping[str, pd.DatetimeIndex], *, start, end,
                    as_of) -> dict:
    """Coverage of each symbol's bar dates against the actual NYSE sessions in
    ``[start, end]``.

    Returns ``{"window": [start, end], "ok": bool, "symbols": {sym: {...}},
    "missing_by_date": {date: [symbols lacking it]}}`` where each symbol entry
    has ``first``, ``last``, ``n_in_window``, ``missing`` (actual sessions in
    the window with no bar), ``extra`` (bars in the window on a closed day) and
    ``live_partial_bar`` (the symbol's last bar is dated today and the ET
    clock has not reached that day's 16:00 close). ``ok`` is False if any
    symbol has any of missing / extra / live_partial_bar. ``settled_through``
    is the last calendar date whose session had settled at ``as_of`` (today
    once the ET close has passed, else yesterday); ``last_complete_session``
    reads it."""
    if not indices:
        raise ValueError("coverage_report needs at least one symbol")
    start_ts, end_ts = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if end_ts < start_ts:
        raise ValueError("end must not precede start")
    from mining.batch5.treasury_month_end_index_demand_overlay import exchange_sessions
    actual = exchange_sessions(start_ts, end_ts)
    now_et = _as_of_et(as_of)
    # a session is only EXPECTED to have a bar once it has settled (its ET
    # close has passed); today before 16:00 and every later date are not holes
    today = now_et.tz_localize(None).normalize()
    last_settled = today if now_et >= pd.Timestamp(
        f"{today:%Y-%m-%d} {MARKET_CLOSE_HOUR_ET}:00", tz=_ET) else today - pd.Timedelta(days=1)
    settled = actual[actual <= last_settled]

    symbols, missing_by_date = {}, {}
    for symbol in sorted(indices):
        idx = _normalized_index(indices[symbol])
        in_window = idx[(idx >= start_ts) & (idx <= end_ts)]
        missing = settled.difference(in_window)
        extra = in_window.difference(actual)
        last = idx.max() if len(idx) else None
        live = False
        if last is not None and last.normalize() == now_et.tz_localize(None).normalize():
            close = pd.Timestamp(f"{last:%Y-%m-%d} {MARKET_CLOSE_HOUR_ET}:00", tz=_ET)
            live = now_et < close
        symbols[symbol] = {
            "first": f"{idx.min():%Y-%m-%d}" if len(idx) else None,
            "last": f"{last:%Y-%m-%d}" if last is not None else None,
            "n_in_window": int(len(in_window.unique())),
            "missing": _iso(missing),
            "extra": _iso(extra.unique()),
            "live_partial_bar": bool(live),
        }
        for stamp in missing:
            missing_by_date.setdefault(f"{stamp:%Y-%m-%d}", []).append(symbol)
    ok = not any(s["missing"] or s["extra"] or s["live_partial_bar"]
                 for s in symbols.values())
    return {"window": [f"{start_ts:%Y-%m-%d}", f"{end_ts:%Y-%m-%d}"], "ok": ok,
            "symbols": symbols,
            "missing_by_date": {d: sorted(v) for d, v in missing_by_date.items()},
            "settled_through": f"{last_settled:%Y-%m-%d}"}


def last_complete_session(report: Mapping):
    """The last actual NYSE session, as an ISO date, that a ``coverage_report``
    shows complete for every symbol, or ``None`` if the window holds none.

    Complete means: a real session, strictly before the FIRST hole across all
    symbols (a symbol's missing session, or any bar on a closed day), and
    settled -- a still-forming bar (today before the ET close) never counts.
    With no holes it is the last settled session of the window. ``None`` when
    the first hole leaves no session before it (a hole at or before the window
    start) or nothing in the window has settled yet.

    It only READS the report and names a date; it fills, patches and repairs
    nothing. The date is complete only within the window that was checked."""
    try:
        start, end = report["window"]
        settled_through = report["settled_through"]
        symbols = report["symbols"]
    except KeyError as exc:
        raise ValueError(
            f"report lacks {exc}; pass the dict coverage_report returns") from exc
    start_ts = pd.Timestamp(start).normalize()
    bound = min(pd.Timestamp(end).normalize(),
                pd.Timestamp(settled_through).normalize())
    if bound < start_ts:
        return None
    from mining.batch5.treasury_month_end_index_demand_overlay import exchange_sessions
    candidates = exchange_sessions(start_ts, bound)
    holes = [pd.Timestamp(d) for row in symbols.values()
             for d in (*row["missing"], *row["extra"])]
    if holes:
        candidates = candidates[candidates < min(holes)]
    return f"{candidates.max():%Y-%m-%d}" if len(candidates) else None
