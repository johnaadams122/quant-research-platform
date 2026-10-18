"""Benchmark-plus-overlay account builder, LONG-ONLY margin-financed overlay.

Pure functions only: no network, no ledger, no clock. Built for
``treasury_month_end_index_demand_overlay`` and written as the long analogue
of ``mining.beta_neutral_overlay`` (which builds a short overlay with a hedge
leg); read that module's docstring for why a benchmark-plus-overlay account,
rather than a rotation out of the benchmark, is the only account construction
whose ABI-1 score can isolate a premium earned in a small part of the calendar.

Why an overlay and not a rotation
---------------------------------
ABI-1 (``mining.account_relative``) scores the ACCOUNT's daily log excess over
SPY buy-and-hold. An account that swaps SPY for a bond fund for a few days a
month scores roughly minus SPY's own return on those days, so the bond premium
it is meant to test is buried under an equity return it gave up. The account
built here instead holds SPY ITSELF at all times (the identical cost-adjusted
benchmark series the evaluator subtracts) and, inside scheduled windows only,
adds a long position in the overlay legs funded by borrowing:

    account return  = r_spy + overlay
    excess over SPY = overlay        (to first order; ABI-1 scores the LOG excess)

There is no hedge leg and no short leg: the overlay is long-only, gross
exposure is ``1 + overlay_notional`` (Reg-T feasible while that is at most 2).

Accounting conventions
----------------------
* Windows are supplied as ``(entry, exit)`` pairs of session dates; the caller
  owns the calendar (for the production family it is the exchange's SCHEDULED
  month-end schedule, fixed in advance, never read off the data).
* The account calendar is the legs' common span (``common_calendar``): the
  leading edge follows the latest-listing leg, but the legs and SPY must END on
  the same session and share identical sessions inside the span. Unequal
  trailing coverage and interior mismatches raise ``CalendarCoverageError``
  (a ``ValueError`` with a ``reason_code`` and the dates) instead of being
  trimmed or patched; whether every actual EXCHANGE session is present is the
  caller's check (the candidate module's ``validate_session_coverage``).
* ENTRY AT THE CLOSE of the ``entry`` session, EXIT AT THE CLOSE of the ``exit``
  session ("ex-ante-close convention": the window dates are known long before
  the trade). The overlay therefore earns the close-to-close move of every
  session AFTER ``entry`` through ``exit``, and nothing on the entry session.
  The engine's own convention (a signal fills at the next session's Open) is
  available ONLY as ``build_next_open_sensitivity_curve``, a report-only
  disclosure that is never scored; the scored builder has no fill argument.
* Sizing: the overlay legs are opened with total notional
  ``overlay_notional`` times the account's equity AT THE ENTRY CLOSE (known at
  decision time), split into equal weights across the legs, at fixed share
  counts for the whole window. Daily overlay P&L is ``sum(n_i * dP_i)`` from
  the actual price path (weights drift), never re-normalised.
* Funding: the whole overlay notional is borrowed. Financing accrues on every
  window session as ``principal * annual_rate * elapsed_calendar_days / 365``
  where ``annual_rate`` is the rate published STRICTLY BEFORE that session
  (see ``causal_financing_rate``) and the elapsed days are the calendar days
  since the previous session in the account calendar, so a Monday carries the
  weekend (the same calendar-day accrual ``account_relative`` uses for cash).
  NO spread over the bill rate is added: the pipeline has no
  margin-rate source and inventing one is not allowed, so a real margin rate
  (which adds a spread to the bill rate) is NOT modelled and this FLATTERS the
  strategy. A negative bill print is applied with its sign intact, exactly as
  the cash producers do.
* Costs: each leg is charged a full round trip (``cost_bps`` is already a round
  trip: half the quoted spread at entry plus half at exit) on the exit day, as
  a fraction of the sizing equity, weighted by the leg's share of the overlay.
* Account daily return is ``core_t + overlay_dollars_t / E_{t-1}`` where
  ``core_t`` is the benchmark's OWN cost-adjusted daily return (so the account
  equals the benchmark series bit for bit outside the windows) and ``E`` is the
  account's pre-tax equity, compounded daily (overlay P&L is added to equity
  and held in SPY).
* A window whose scheduled exit lies after the last session of the data is
  closed at the last close (``reason == "end_of_data"``, the engine's own
  path); a window entered on the very last bar opens nothing.

LIMITATIONS (disclosed, each of the first three FLATTERS the strategy)
----------------------------------------------------------------------
Margin-rate spread over the bill rate; short-term tax on the overlay's gains is
modelled by ABI-1's realization walk on ``trades_df`` (net of financing, i.e.
interest treated as fully deductible against the overlay's gain); Reg-T margin
calls and any borrow-availability limit; and the netted tax walk's entry-equity
identity assumes the account grows only by the trade, while here the SPY core
moves too (the driver measures the size of that approximation).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from . import account_relative
from .benchmark_relative import cost_adjusted_benchmark_daily_returns

TRADES_COLUMNS = ("entry_date", "exit_date", "held", "reason", "size",
                  "gross_ret", "financing_ret", "cost_ret", "net_ret")

# Gross exposure is 1 (the SPY core) + overlay_notional; Reg-T initial margin
# (50 percent of long value) is met while that is at most 2.
MAX_OVERLAY_NOTIONAL = 1.0


# ------------------------------------------------------------- validation

def _validated_frame(frame, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name}: expected a non-empty OHLC DataFrame")
    for column in ("Open", "Close"):
        if column not in frame.columns:
            raise ValueError(f"{name}: frame is missing {column}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: frame needs a DatetimeIndex")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError(f"{name}: index must be sorted and unique")
    values = frame[["Open", "Close"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError(f"{name}: Open/Close must be finite and positive")
    return frame


def _validated_scalar(value, name: str, *, low: float, high: float,
                      inclusive_low: bool) -> float:
    value = float(value)
    ok = np.isfinite(value) and (value >= low if inclusive_low else value > low)
    if not ok or value > high:
        raise ValueError(f"{name} must be finite and within "
                         f"{'[' if inclusive_low else '('}{low}, {high}]; "
                         f"got {value}")
    return value


class CalendarCoverageError(ValueError):
    """A data calendar this family refuses to score. A ``ValueError`` (every
    existing ``except ValueError`` still catches it) that also carries a
    machine-readable ``reason_code`` and the dates involved, so a driver can
    report a clean refusal instead of a traceback:

    * ``unequal_trailing_coverage`` -- the legs and SPY do not all end on the
      same session (the account would silently end early);
    * ``calendar_mismatch`` -- the series disagree INSIDE the legs' common span;
    * ``missing_exchange_session`` / ``bar_on_closed_day`` -- raised by the
      candidate module's exchange-session check.

    ``missing_dates`` are sessions a series lacks, ``extra_dates`` bars on days
    the exchange was closed (both ISO strings, sorted, complete)."""

    def __init__(self, reason_code: str, message: str, *, missing_dates=(),
                 extra_dates=()):
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.missing_dates = _iso_dates(missing_dates)
        self.extra_dates = _iso_dates(extra_dates)


def _iso_dates(dates) -> list:
    return sorted({pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates})


def _listed(dates, limit: int = 20) -> str:
    """Dates for an error message: the first ``limit`` and a count of the rest."""
    stamps = _iso_dates(dates)
    text = ", ".join(stamps[:limit])
    return text + (f" (and {len(stamps) - limit} more)" if len(stamps) > limit else "")


def common_calendar(spy_frame, leg_frames: Mapping) -> pd.DatetimeIndex:
    """The calendar the account lives on: the legs' common span, on which EVERY
    leg must have an identical session list and SPY must match it exactly.

    LEADING edge: the span starts at the latest first session (a later-listing
    leg, e.g. AGG, legitimately shortens it) and SPY's own longer history is
    sliced to it. TRAILING edge: the legs and SPY must all END on the same
    session; unequal trailing coverage (a vendor missing the last bar of one
    series) is refused with ``unequal_trailing_coverage`` rather than trimmed,
    because trimming would silently end the account a session early. An
    interior mismatch between same-exchange instruments is a data defect and
    fails closed (``calendar_mismatch``) rather than being patched by
    intersecting calendars. ``CalendarCoverageError`` is a ``ValueError``."""
    spy = _validated_frame(spy_frame, "spy_frame")
    if not leg_frames:
        raise ValueError("leg_frames must name at least one overlay leg")
    legs = {sym: _validated_frame(frame, sym) for sym, frame in leg_frames.items()}
    frames = {**legs, "SPY": spy}
    ends = {sym: frame.index[-1] for sym, frame in frames.items()}
    if len(set(ends.values())) > 1:
        latest, earliest = max(ends.values()), min(ends.values())
        lacking = sorted({d for frame in frames.values()
                          for d in frame.index[frame.index > earliest]})
        raise CalendarCoverageError(
            "unequal_trailing_coverage",
            "the overlay legs and SPY must all end on the same session, but "
            + ", ".join(f"{sym} ends {end:%Y-%m-%d}" for sym, end in ends.items())
            + f"; {', '.join(sym for sym, end in ends.items() if end < latest)} "
            f"lack the trailing session(s) {_listed(lacking)}; refusing to trim "
            "the account to the shortest series (refetch the history, or wait "
            "for the vendor to repair it)",
            missing_dates=lacking)
    start = max(frame.index[0] for frame in legs.values())
    end = min(frame.index[-1] for frame in legs.values())
    if start > end:
        raise ValueError("the overlay legs share no common span")
    spans = {sym: frame.loc[start:end].index for sym, frame in legs.items()}
    first_symbol = next(iter(spans))
    first = spans[first_symbol]
    for sym, span in spans.items():
        if not span.equals(first):
            raise CalendarCoverageError(
                "calendar_mismatch",
                f"{sym}: the overlay legs must share an identical calendar "
                f"inside their common span ({start.date()}..{end.date()}); "
                f"{_first_differences(span, sym, first, first_symbol)}; "
                "refusing to patch a calendar mismatch between same-exchange "
                "instruments (refetch the history)",
                missing_dates=span.symmetric_difference(first))
    spy_span = spy.loc[start:end].index
    if not spy_span.equals(first):
        raise CalendarCoverageError(
            "calendar_mismatch",
            "spy_frame: the SPY sessions inside the legs' common span "
            f"({start.date()}..{end.date()}) must match the leg calendar "
            f"exactly; {_first_differences(spy_span, 'SPY', first, first_symbol)}; "
            "refusing to patch a calendar mismatch (refetch the history)",
            missing_dates=spy_span.symmetric_difference(first))
    return first


def _first_differences(left: pd.DatetimeIndex, left_name: str,
                       right: pd.DatetimeIndex, right_name: str,
                       limit: int = 3) -> str:
    """Names the first sessions present on one side only, so a vendor hole can
    be identified (and refetched) instead of guessed at."""
    only_left = left.difference(right)[:limit]
    only_right = right.difference(left)[:limit]
    parts = []
    if len(only_left):
        parts.append(f"{left_name} has but {right_name} lacks "
                     + ", ".join(d.strftime("%Y-%m-%d") for d in only_left))
    if len(only_right):
        parts.append(f"{right_name} has but {left_name} lacks "
                     + ", ".join(d.strftime("%Y-%m-%d") for d in only_right))
    return "; ".join(parts) or "the session lists differ"


# --------------------------------------------------------- financing rate

def causal_financing_rate(irx_close_percent, index) -> pd.Series:
    """Annualized borrowing rate applicable to each session of ``index``: the
    13-week bill's coupon-equivalent yield computed from the ^IRX close of the
    last session STRICTLY BEFORE it (union-then-shift, the same causal rule
    ``account_relative.daily_cash_rate`` uses). ``irx_close_percent`` is the raw
    ^IRX Close in percent (3.6 means 3.6 percent, a bank DISCOUNT quote); NaN
    where no earlier print exists. No spread is added. A negative print passes
    through with its sign."""
    irx = pd.Series(irx_close_percent, dtype=float)
    irx = irx[~irx.index.duplicated(keep="last")].sort_index().dropna()
    idx = pd.DatetimeIndex(index)
    prior = irx.reindex(irx.index.union(idx)).ffill().shift(1).reindex(idx)
    annual = account_relative.coupon_equivalent_yield(
        prior.to_numpy(dtype=float) / 100.0)
    return pd.Series(np.asarray(annual, dtype=float), index=idx,
                     name="financing_rate")


# --------------------------------------------------------------- builder

def _parse_windows(windows: Sequence, idx: pd.DatetimeIndex) -> list:
    n = len(idx)
    parsed = []
    for k, pair in enumerate(windows):
        try:
            entry, exit_ = pair
        except (TypeError, ValueError):
            raise ValueError(f"window {k}: expected an (entry, exit) pair") from None
        entry = pd.Timestamp(entry).normalize()
        exit_ = pd.Timestamp(exit_).normalize()
        entry_pos = int(idx.get_indexer([entry])[0])
        if entry_pos < 0:
            raise ValueError(f"window {k}: entry {entry.date()} is not a "
                             "session of the account calendar")
        if not entry < exit_:
            raise ValueError(f"window {k}: entry {entry.date()} must be "
                             f"strictly before its exit {exit_.date()}")
        if exit_ > idx[-1]:
            exit_pos, reason = n - 1, "end_of_data"
        else:
            exit_pos, reason = int(idx.get_indexer([exit_])[0]), "window"
            if exit_pos < 0:
                raise ValueError(f"window {k}: exit {exit_.date()} is not a "
                                 "session of the account calendar")
        if parsed and entry_pos <= parsed[-1]["exit_pos"]:
            raise ValueError(
                "windows must be strictly ascending and must not overlap; "
                f"window {k} enters at {entry.date()} before the previous "
                f"window exits at {idx[parsed[-1]['exit_pos']].date()}")
        parsed.append({"entry_pos": entry_pos, "exit_pos": exit_pos,
                       "reason": reason})
    return parsed


def _build(spy_frame, leg_frames, windows, *, leg_cost_bps, spy_entry_cost_bps,
           financing_rate_annual, overlay_notional,
           entry_at_next_open: bool) -> dict:
    idx = common_calendar(spy_frame, leg_frames)
    n = len(idx)
    symbols = list(leg_frames)
    x = _validated_scalar(overlay_notional, "overlay_notional", low=0.0,
                          high=MAX_OVERLAY_NOTIONAL, inclusive_low=False)
    entry_cost = _validated_scalar(spy_entry_cost_bps, "spy_entry_cost_bps",
                                   low=0.0, high=1e4, inclusive_low=True)
    if not isinstance(leg_cost_bps, Mapping) or set(leg_cost_bps) != set(symbols):
        raise ValueError("leg_cost_bps must map exactly the overlay legs "
                         f"{symbols} to a round-trip cost in bps")
    costs = np.array([_validated_scalar(leg_cost_bps[s], "leg_cost_bps",
                                        low=0.0, high=1e4, inclusive_low=True)
                      for s in symbols]) / 1e4
    weights = np.full(len(symbols), 1.0 / len(symbols))
    parsed = _parse_windows(windows, idx)

    core_series = cost_adjusted_benchmark_daily_returns(
        spy_frame, entry_cost_bps=entry_cost).reindex(idx)
    core = core_series.to_numpy(dtype=float)
    if not np.isfinite(core).all():
        raise ValueError("benchmark core returns must be finite on every "
                         "account session")

    fin = pd.Series(financing_rate_annual, dtype=float).reindex(idx) \
        .to_numpy(dtype=float)
    close = np.vstack([leg_frames[s]["Close"].loc[idx].to_numpy(dtype=float)
                       for s in symbols])
    open_ = np.vstack([leg_frames[s]["Open"].loc[idx].to_numpy(dtype=float)
                       for s in symbols])
    elapsed = np.concatenate(
        [[1.0], np.diff(idx.values.astype("datetime64[D]")).astype(float)])
    days_per_year = float(account_relative.CALENDAR_DAYS_PER_YEAR)

    open_after, open_before = {}, {}
    for w in parsed:
        if w["entry_pos"] >= n - 1:
            continue                      # nothing left to earn: opens nothing
        if entry_at_next_open:
            open_before[w["entry_pos"] + 1] = w
        else:
            open_after[w["entry_pos"]] = w

    strategy = np.empty(n, dtype=float)
    overlay_frac = np.zeros(n, dtype=float)
    mask = np.zeros(n, dtype=bool)
    trades = []
    equity = 1.0
    seg = None

    def _open(w, size, prices, first_pos):
        return {
            "w": w, "size": size, "first_pos": first_pos,
            "entry_px": prices.copy(),
            "shares": (x * weights * size) / prices,
            "principal": x * size,
            "cost": size * x * float((weights * costs).sum()),
            "price_pnl": 0.0, "fin": 0.0,
        }

    for t in range(n):
        e_prev = equity
        if not np.isfinite(e_prev) or e_prev <= 0.0:
            raise ValueError(f"account equity is {e_prev} before "
                             f"{idx[t].date()}; the overlay wiped the account")
        if t in open_before:
            seg = _open(open_before[t], e_prev, open_[:, t], t)
        overlay = 0.0
        if seg is not None:
            reference = seg["entry_px"] if t == seg["first_pos"] else close[:, t - 1]
            price_pnl = float((seg["shares"] * (close[:, t] - reference)).sum())
            rate = fin[t]
            if not np.isfinite(rate):
                raise ValueError(
                    f"financing rate is not finite on {idx[t].date()}, a "
                    "session the overlay is financed; refusing to assume a "
                    "zero borrowing cost")
            days = 1.0 if (entry_at_next_open and t == seg["first_pos"]) \
                else elapsed[t]
            fin_cost = seg["principal"] * rate * days / days_per_year
            overlay = price_pnl - fin_cost
            seg["price_pnl"] += price_pnl
            seg["fin"] += fin_cost
            mask[t] = True
            if t == seg["w"]["exit_pos"]:
                overlay -= seg["cost"]
                size = seg["size"]
                entry_pos = seg["w"]["entry_pos"]
                trades.append({
                    "entry_date": idx[seg["first_pos"] if entry_at_next_open
                                      else entry_pos],
                    "exit_date": idx[t],
                    "held": int(t - entry_pos),
                    "reason": seg["w"]["reason"],
                    "size": size,
                    "gross_ret": seg["price_pnl"] / size,
                    "financing_ret": seg["fin"] / size,
                    "cost_ret": seg["cost"] / size,
                    "net_ret": (seg["price_pnl"] - seg["fin"] - seg["cost"]) / size,
                })
                seg = None
        daily = core[t] + overlay / e_prev
        if not np.isfinite(daily) or daily <= -1.0:
            raise ValueError(f"account return {daily} at {idx[t].date()} is "
                             "not a valid simple return")
        strategy[t] = daily
        overlay_frac[t] = overlay / e_prev
        equity = e_prev * (1.0 + daily)
        if t in open_after:
            seg = _open(open_after[t], equity, close[:, t], t + 1)

    if trades:
        trades_df = pd.DataFrame(trades, columns=list(TRADES_COLUMNS))
    else:
        trades_df = pd.DataFrame({
            "entry_date": pd.Series(dtype="datetime64[ns]"),
            "exit_date": pd.Series(dtype="datetime64[ns]"),
            "held": pd.Series(dtype="int64"),
            "reason": pd.Series(dtype="object"),
            "size": pd.Series(dtype="float64"),
            "gross_ret": pd.Series(dtype="float64"),
            "financing_ret": pd.Series(dtype="float64"),
            "cost_ret": pd.Series(dtype="float64"),
            "net_ret": pd.Series(dtype="float64"),
        })[list(TRADES_COLUMNS)]
    return {
        "strategy_daily": pd.Series(strategy, index=idx,
                                    name="long_overlay_account"),
        "in_market": pd.Series(True, index=idx, name="in_market"),
        "trades_df": trades_df,
        # Risk-basis rule: ABI-1 measures this account's risk layer on the
        # overlay's own excess over SPY (`abi1_driver.evaluate_family` reads
        # this key), not on the whole SPY-core account.
        "account_kind": account_relative.ACCOUNT_KIND_BENCHMARK_PLUS_OVERLAY,
        "core_daily": core_series.rename("core"),
        "overlay_daily": pd.Series(overlay_frac, index=idx, name="overlay"),
        "overlay_mask": pd.Series(mask, index=idx, name="overlay_mask"),
        "financing_rate": pd.Series(fin, index=idx, name="financing_rate"),
    }


def build_long_overlay_curve(spy_frame, leg_frames, windows, *, leg_cost_bps,
                             spy_entry_cost_bps, financing_rate_annual,
                             overlay_notional) -> dict:
    """The SCORED account curve (entry at the close of each window's entry
    session, exit at the close of its exit session), in the
    ``{"strategy_daily", "in_market", "trades_df"}`` shape
    ``abi1_driver.evaluate_family`` accepts from any builder, plus report-only
    extras (``core_daily``, ``overlay_daily``, ``overlay_mask``,
    ``financing_rate``).

    ``spy_frame`` is the benchmark's FULL history (a superset of the legs'
    calendar); ``spy_entry_cost_bps`` is the benchmark's own already-halved
    entry cost, the identical figure the ABI-1 evaluator charges the
    benchmark, so the account's SPY core equals the benchmark series exactly.
    ``leg_cost_bps`` maps every leg to its ROUND-TRIP cost in bps.
    ``financing_rate_annual`` is the borrowing rate applicable to each session
    (``causal_financing_rate``); it is only required to be finite on sessions
    the overlay is financed."""
    return _build(spy_frame, leg_frames, windows, leg_cost_bps=leg_cost_bps,
                  spy_entry_cost_bps=spy_entry_cost_bps,
                  financing_rate_annual=financing_rate_annual,
                  overlay_notional=overlay_notional, entry_at_next_open=False)


def build_next_open_sensitivity_curve(spy_frame, leg_frames, windows, *,
                                      leg_cost_bps, spy_entry_cost_bps,
                                      financing_rate_annual,
                                      overlay_notional) -> dict:
    """REPORT-ONLY disclosed sensitivity: the same account with the overlay
    bought at the OPEN of the first window session (the engine's next-open
    fill of a signal raised at the entry close) instead of the entry close.
    Financing for that first session accrues one calendar day (the overlay is
    only borrowed intraday). Never scored, never written to a ledger, and
    deliberately a separate function so the scored builder cannot be switched
    to it."""
    return _build(spy_frame, leg_frames, windows, leg_cost_bps=leg_cost_bps,
                  spy_entry_cost_bps=spy_entry_cost_bps,
                  financing_rate_annual=financing_rate_annual,
                  overlay_notional=overlay_notional, entry_at_next_open=True)
