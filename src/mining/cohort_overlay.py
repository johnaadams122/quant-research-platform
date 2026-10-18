"""Benchmark-plus-overlay account builder whose overlay is a monthly SELECTED
COHORT of names minus the equal-weight basket of ALL ELIGIBLE names, with the
seeded permutation placebo and the zero-signal equal-weight control curve.

Pure functions only: no network, no ledger, no clock, no file access. Written
for ``residual_momentum_ff3proxy_top50_monthly`` and generic on purpose -- the
cohort is chosen by a caller-supplied per-month ranking, so a high-volume-shock
or low-asset-growth family can reuse it unchanged.

RESEARCH CONTROL, NOT AN INVESTABLE ACCOUNT
-------------------------------------------
The short leg of this overlay is the equal-weight basket of ALL eligible names
(hundreds of stocks, re-formed monthly). It exists to cancel the LEVEL of the
survivorship and equal-weight-tilt baseline that a panel of today's index
members carries into any long-only cohort (a no-signal equal-weight basket of
such a panel can score well against SPY on its own). It does NOT cancel
rank-conditional survivorship (names the ranking favoured that later failed or
joined the index later), and the permutation placebo, drawing from the same
eligible set, shares that limitation and cannot test it. It is a measurement
device: nobody could hold it, and a pass here does not certify an investable
long-only strategy -- that would need its own design and its own trial.

Why a benchmark-plus-overlay account
------------------------------------
ABI-1 (``mining.account_relative``) scores the account's daily excess over SPY.
The account built here holds SPY ITSELF at all times (the identical
cost-adjusted benchmark series the evaluator subtracts) and, inside scheduled
windows, adds a dollar-neutral overlay:

    account return  = r_spy + overlay
    overlay         = x * ( EW(selected cohort) - EW(all eligible names) )

so the scored excess is the overlay only, and outside the windows the account
equals the benchmark bit for bit. The risk-basis rule applies: the curve declares
``account_kind == "benchmark_plus_overlay"`` and ABI-1's four risk refusals are
measured on the overlay-only excess account.

Accounting conventions (mirrors ``mining.long_overlay``'s next-open path)
-------------------------------------------------------------------------
* A window is ``(entry, exit)``: ENTRY AT THE OPEN of the ``entry`` session
  (the first session of the holding month), EXIT AT THE CLOSE of the ``exit``
  session. The overlay therefore earns the open-to-close move of the entry
  session and close-to-close moves afterwards; the overnight gap INTO the entry
  session is not earned (the forced round trip skips the month-boundary gap).
  The ``decision`` session is the session immediately before ``entry``; it is
  where the signal was known and where the sizing equity is read.
* The account calendar runs from the first window's decision session to the
  last window's exit, on the SPY frame's own sessions. Sessions outside the
  windows (the first decision session, and any skipped month) carry the plain
  benchmark return, exactly.
* Sizing: each leg is opened at ``overlay_notional`` times the account equity
  at the DECISION close, split equally across the leg's names, in fixed share
  counts for the whole window (weights drift with prices; nothing is
  rebalanced inside a window).
* Financing: only the LONG leg is borrowed, at the bare bill rate published
  strictly before each session (``long_overlay.causal_financing_rate``), on
  calendar days over 365, one calendar day on the first window session. The
  short leg is credited NO rebate and charged NO borrow fee: not crediting the
  rebate is conservative, omitting the fee flatters, and a real margin
  spread over the bill rate is not modelled either (the pipeline has no
  margin-rate source and inventing one is not allowed).
* Costs: each name is charged a full round trip (``cost_bps_by_symbol`` is
  already a round trip: half the quoted spread in, half out) on the exit day,
  weighted by its share of its leg, on BOTH legs: the selected cohort's
  average cost plus the eligible basket's average cost, times
  ``overlay_notional``, as a fraction of the sizing equity.
* Fail closed, never patch: any eligible name (selected or not) lacking the
  Open on the entry session or a Close on any window session raises
  ``CohortDataError`` naming the tickers and dates; nothing is dropped,
  shifted or forward-filled.
* Selection ties: ``select_top_n`` ranks the highest scores first and breaks
  ties by ticker ascending, where scores within ``RANK_REL_TOL`` (a relative
  1e-12) of their neighbour are ONE tie: byte-identical inputs can score a last
  bit apart (``np.linalg.lstsq`` is not column-position invariant), and the
  larger ticker must not take a slot on that rounding accident.
* Account identity: ``require_account_is_benchmark_plus_overlay`` recomputes,
  on EVERY session, that the account equals the benchmark plus the overlay and
  that the core equals the benchmark, because the shared off-window guard reads
  only the sessions the overlay is off (one, on a contiguous monthly schedule).

LIMITATIONS (each of the first three FLATTERS the strategy)
-----------------------------------------------------------
No short borrow fee; no margin spread over the bill rate; no Reg-T margin call
or borrow-availability limit. Not labelled as flattering: short-term tax on the
overlay is ABI-1's realization walk on the trade list, and the tax identity it
relies on (a dollar-attribution approximation, because the SPY core moves too)
is approximate here in an unmeasured direction (the driver measures its ratio
range). The eligible universe passed in is the caller's: if it is today's index
members it is survivorship-biased, which the short basket cancels only at the
LEVEL, and only to the extent both legs share the same survivors; rank-
conditional survivorship is not cancelled and the placebo cannot test it.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import account_relative
from .benchmark_relative import cost_adjusted_benchmark_daily_returns
from .long_overlay import CalendarCoverageError, causal_financing_rate  # noqa: F401  (re-exported)

# Gross exposure of the account is 1 (SPY core) + x (long leg) + x (short leg);
# Reg-T initial margin (half of gross) fits the equity while 0.5 * (1 + 2x) <= 1.
MAX_COHORT_OVERLAY_NOTIONAL = 0.5

# The permutation placebo is report-only but must carry enough draws for a
# 95th-percentile read to mean something.
MIN_PLACEBO_DRAWS = 300

# Two scores within this RELATIVE distance (about twelve significant digits, a few thousand ulps) are
# one tie in ``select_top_n``: the ticker, not a rounding accident in the last bit, decides between them.
RANK_REL_TOL = 1e-12

TRADES_COLUMNS = ("entry_date", "exit_date", "held", "reason", "size",
                  "gross_ret", "financing_ret", "cost_ret", "net_ret",
                  "n_selected", "n_eligible")

PLACEBO_STATISTIC = "annualized_sharpe_of_overlay_daily_arithmetic_excess"
_TRADING_DAYS_PER_YEAR = 252.0


class CohortDataError(ValueError):
    """A data defect this builder refuses to score (a ``ValueError`` with a
    machine-readable ``reason_code`` plus the tickers and ISO dates involved):

    * ``missing_member_frame`` -- a window names a ticker with no price frame;
    * ``missing_member_price`` -- a named ticker lacks the Open on the entry
      session or a Close on some window session;
    * ``missing_member_cost`` -- a named ticker has no round-trip cost.
    """

    def __init__(self, reason_code: str, message: str, *, tickers=(), dates=()):
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.tickers = sorted({str(t) for t in tickers})
        self.dates = sorted({pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates})


@dataclass(frozen=True)
class CohortWindow:
    """One holding window. ``entry``/``exit`` are the first and last sessions
    of the hold; ``selected`` is the cohort (long leg), ``eligible`` the full
    eligible set (the equal-weight short leg) and ``selected`` must be a
    subset of it. ``decision`` (optional) is the session where the signal was
    known: when given it must be the session immediately before ``entry``."""
    entry: pd.Timestamp
    exit: pd.Timestamp
    selected: tuple
    eligible: tuple
    decision: Optional[pd.Timestamp] = None


# ------------------------------------------------------------------ selection

def select_top_n(scores, n: int, *, rel_tol: float = RANK_REL_TOL) -> tuple:
    """The ``n`` highest finite scores, ties broken by ticker ascending. Fewer
    than ``n`` finite scores returns what exists (never pads).

    A TIE is a run of scores, in descending order, each within ``rel_tol`` of
    the one before it (relative to the larger magnitude of the pair): about
    twelve significant digits by default. Two names with byte-identical inputs
    can score a last bit apart (``np.linalg.lstsq`` is not column-position
    invariant), so ranking on the raw floats would let the LARGER ticker take a
    slot on a rounding accident; inside a tie class the ticker decides. Rounding
    to significant digits was not used because a pair that straddles a rounding
    boundary would be split; comparing neighbours cannot split a pair of
    near-equal scores (any score between them is within the tolerance too)."""
    series = pd.Series(scores, dtype=float)
    series = series[np.isfinite(series.to_numpy(dtype=float))]
    ordered = sorted(zip(series.to_numpy(dtype=float).tolist(), map(str, series.index)),
                     key=lambda pair: (-pair[0], pair[1]))
    keyed, head, previous = [], None, None
    for value, ticker in ordered:
        if previous is None or not (previous - value) <= rel_tol * max(abs(previous), abs(value)):
            head = value                                    # a new tie class starts at its highest score
        keyed.append((-head, ticker))
        previous = value
    keyed.sort()
    return tuple(ticker for _, ticker in keyed[:int(n)])


def plan_cohort_windows(month_windows, score_fn: Callable, n_select: int) -> dict:
    """Turn ``(decision, entry, exit)`` month windows into ``CohortWindow``s.

    ``score_fn(decision)`` is the caller's ranking: a Series of scores indexed
    by ticker, computed from data on or before ``decision`` only (this function
    cannot enforce that; the family's own test does). The ELIGIBLE set is the
    names with a finite score; the cohort is the top ``n_select`` of them. A
    month with fewer than ``n_select`` eligible names holds nothing and is
    listed in ``skipped`` as ``(decision, n_eligible)``."""
    if isinstance(n_select, bool) or not isinstance(n_select, int) or n_select < 1:
        raise ValueError(f"n_select must be a positive integer; got {n_select!r}")
    windows, skipped = [], []
    for decision, entry, exit_ in month_windows:
        scores = pd.Series(score_fn(decision), dtype=float)
        finite = scores[np.isfinite(scores.to_numpy(dtype=float))]
        eligible = tuple(sorted(map(str, finite.index)))
        if len(eligible) < n_select:
            skipped.append((pd.Timestamp(decision), len(eligible)))
            continue
        windows.append(CohortWindow(
            entry=pd.Timestamp(entry), exit=pd.Timestamp(exit_),
            selected=select_top_n(finite, n_select), eligible=eligible,
            decision=pd.Timestamp(decision)))
    return {"windows": windows, "skipped": skipped}


def month_windows(sessions, first_decision, last_decision) -> list:
    """Ex-ante scheduled MONTHLY windows from an explicit list of exchange
    sessions (the caller's calendar; never inferred from any price data):
    ``(decision, entry, exit)`` for every month M whose last session lies in
    ``[first_decision, last_decision]`` -- the decision is the last session of
    M, the entry the first session of M+1, the exit the last session of M+1.
    ``sessions`` must be sorted, unique, and extend past the end of the month
    after the last decision (a whole following month is required: a partial one
    would silently misplace the exit); otherwise ``ValueError``."""
    idx = pd.DatetimeIndex(sessions)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    idx = idx.normalize()
    if len(idx) == 0 or not idx.is_monotonic_increasing or idx.has_duplicates:
        raise ValueError("sessions must be a non-empty, sorted, unique list of exchange sessions")
    start, end = _stamp(first_decision), _stamp(last_decision)
    grouped = pd.Series(idx, index=idx).groupby(idx.to_period("M"))
    last, first = grouped.max(), grouped.min()
    out = []
    for period in last.index:
        decision = last[period]
        if decision < start or decision > end:
            continue
        following = period + 1
        if following not in last.index or not idx[-1] > last[following]:
            raise ValueError(f"the sessions must extend past the end of month {following} (needed by the "
                             f"decision {decision.date()}); refusing to place a window on a partial month")
        out.append((decision, first[following], last[following]))
    return out


# ----------------------------------------------------------------- validation

def _validated_spy(frame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("spy_frame: expected a non-empty OHLC DataFrame")
    for column in ("Open", "Close"):
        if column not in frame.columns:
            raise ValueError(f"spy_frame: frame is missing {column}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("spy_frame: frame needs a DatetimeIndex")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("spy_frame: index must be sorted and unique")
    values = frame[["Open", "Close"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("spy_frame: Open/Close must be finite and positive")
    return frame


def _validated_scalar(value, name: str, *, low: float, high: float, inclusive_low: bool) -> float:
    value = float(value)
    ok = np.isfinite(value) and (value >= low if inclusive_low else value > low)
    if not ok or value > high:
        raise ValueError(f"{name} must be finite and within "
                         f"{'[' if inclusive_low else '('}{low}, {high}]; got {value}")
    return value


def _stamp(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize()


class _Panel:
    """Aligned Open/Close matrices for every name any window references, on the
    account calendar, plus per-window caches of the eligible basket (identical
    across placebo draws)."""

    def __init__(self, spy_frame, member_frames: Mapping, windows: Sequence,
                 cost_bps_by_symbol: Mapping):
        spy = _validated_spy(spy_frame)
        if not windows:
            raise ValueError("windows must contain at least one CohortWindow")
        wins = [self._normalized(w) for w in windows]
        spy_entry = spy.index.get_indexer([w.entry for w in wins])
        spy_exit = spy.index.get_indexer([w.exit for w in wins])
        previous_exit = -1
        for k, w in enumerate(wins):
            if spy_entry[k] < 0:
                raise ValueError(f"window {k}: entry {w.entry.date()} is not a session of the SPY frame")
            if spy_exit[k] < 0:
                raise ValueError(f"window {k}: exit {w.exit.date()} is not a session of the SPY frame")
            if not w.entry < w.exit:
                raise ValueError(f"window {k}: entry {w.entry.date()} must be strictly before its exit {w.exit.date()}")
            if spy_entry[k] <= previous_exit:
                raise ValueError("windows must be strictly ascending and must not overlap; "
                                 f"window {k} enters at {w.entry.date()} before the previous "
                                 f"window exits at {spy.index[previous_exit].date()}")
            if spy_entry[k] == 0:
                raise ValueError(f"window {k}: the SPY frame needs a session before the entry "
                                 f"{w.entry.date()} (the decision session that sizes the account)")
            if w.decision is not None and spy.index[spy_entry[k] - 1] != w.decision:
                raise ValueError(f"window {k}: decision {w.decision.date()} must be the session "
                                 f"immediately before the entry {w.entry.date()}")
            previous_exit = int(spy_exit[k])
        offset = int(spy_entry[0]) - 1
        idx = spy.index[offset:int(spy_exit[-1]) + 1]
        self.idx, self.n, self.spy = idx, len(idx), spy
        self.windows = wins

        names = sorted({t for w in wins for t in w.eligible})
        missing = [t for t in names if t not in member_frames]
        if missing:
            raise CohortDataError("missing_member_frame",
                                  f"no price frame for {', '.join(missing[:10])}"
                                  + (f" (and {len(missing) - 10} more)" if len(missing) > 10 else ""),
                                  tickers=missing)
        no_cost = [t for t in names if t not in cost_bps_by_symbol]
        if no_cost:
            raise CohortDataError("missing_member_cost",
                                  f"no round-trip cost for {', '.join(no_cost[:10])}",
                                  tickers=no_cost)
        self.names = names
        self.col = {t: j for j, t in enumerate(names)}
        self.cost = np.array([_validated_scalar(cost_bps_by_symbol[t], "cost_bps_by_symbol",
                                                low=0.0, high=1e4, inclusive_low=True)
                              for t in names], dtype=float) / 1e4
        self.open = np.full((self.n, len(names)), np.nan)
        self.close = np.full((self.n, len(names)), np.nan)
        for t in names:
            frame = member_frames[t]
            if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
                raise ValueError(f"{t}: expected an OHLC DataFrame with a DatetimeIndex")
            for column in ("Open", "Close"):
                if column not in frame.columns:
                    raise ValueError(f"{t}: frame is missing {column}")
            if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
                raise ValueError(f"{t}: index must be sorted and unique")
            values = frame[["Open", "Close"]].to_numpy(dtype=float)
            if np.isinf(values).any() or (values[np.isfinite(values)] <= 0.0).any():
                raise ValueError(f"{t}: Open/Close must be positive (NaN marks a missing bar)")
            self.open[:, self.col[t]] = frame["Open"].reindex(idx).to_numpy(dtype=float)
            self.close[:, self.col[t]] = frame["Close"].reindex(idx).to_numpy(dtype=float)

        self.entry_pos, self.exit_pos, self.sel_cols, self.all_cols = [], [], [], []
        for k, w in enumerate(wins):
            e = int(spy_entry[k]) - offset
            x = int(spy_exit[k]) - offset
            if len(w.selected) == 0:
                raise ValueError(f"window {k}: the selected cohort is empty")
            if len(set(w.selected)) != len(w.selected) or len(set(w.eligible)) != len(w.eligible):
                raise ValueError(f"window {k}: duplicate tickers in the cohort or the eligible set")
            if not set(w.selected) <= set(w.eligible):
                raise ValueError(f"window {k}: the selected cohort must be a subset of the eligible set")
            cols = np.array([self.col[t] for t in w.eligible], dtype=int)
            bad = np.isnan(self.open[e, cols]) | np.isnan(self.close[e:x + 1][:, cols]).any(axis=0)
            if bad.any():
                bad_names = [w.eligible[i] for i in np.flatnonzero(bad)]
                bad_dates = [idx[e + r] for r in range(x - e + 1)
                             if np.isnan(self.close[e + r, cols[bad]]).any()]
                if np.isnan(self.open[e, cols[bad]]).any():
                    bad_dates.append(idx[e])
                raise CohortDataError(
                    "missing_member_price",
                    f"window {k} ({w.entry.date()}..{w.exit.date()}): {len(bad_names)} eligible name(s) "
                    f"lack the entry Open or a window Close: {', '.join(bad_names[:10])}"
                    + (f" (and {len(bad_names) - 10} more)" if len(bad_names) > 10 else "")
                    + "; refusing to drop, shift or fill them",
                    tickers=bad_names, dates=bad_dates)
            self.entry_pos.append(e)
            self.exit_pos.append(x)
            self.all_cols.append(cols)
            self.sel_cols.append(np.array([self.col[t] for t in w.selected], dtype=int))
        self._all_cache: dict = {}
        self.elapsed = np.concatenate(
            [[1.0], np.diff(idx.values.astype("datetime64[D]")).astype(float)])

    @staticmethod
    def _normalized(window):
        if not isinstance(window, CohortWindow):
            raise ValueError("windows must be CohortWindow instances")
        return CohortWindow(entry=_stamp(window.entry), exit=_stamp(window.exit),
                            selected=tuple(window.selected), eligible=tuple(window.eligible),
                            decision=None if window.decision is None else _stamp(window.decision))

    def unit_pnl(self, k: int, cols) -> np.ndarray:
        """Per-session P&L of an equal-weight basket of ``cols`` opened at the
        window's entry Open, per 1 of leg notional, in fixed shares."""
        e, x = self.entry_pos[k], self.exit_pos[k]
        opened = self.open[e, cols]
        closes = self.close[e:x + 1][:, cols]
        reference = np.vstack([opened[None, :], closes[:-1]])
        return ((closes - reference) / opened).mean(axis=1)

    def all_leg(self, k: int):
        cached = self._all_cache.get(k)
        if cached is None:
            cols = self.all_cols[k]
            cached = (self.unit_pnl(k, cols), float(self.cost[cols].mean()))
            self._all_cache[k] = cached
        return cached


# ----------------------------------------------------------------- simulation

def _trades_frame(rows) -> pd.DataFrame:
    if rows:
        return pd.DataFrame(rows, columns=list(TRADES_COLUMNS))
    return pd.DataFrame({
        "entry_date": pd.Series(dtype="datetime64[ns]"),
        "exit_date": pd.Series(dtype="datetime64[ns]"),
        "held": pd.Series(dtype="int64"), "reason": pd.Series(dtype="object"),
        "size": pd.Series(dtype="float64"), "gross_ret": pd.Series(dtype="float64"),
        "financing_ret": pd.Series(dtype="float64"), "cost_ret": pd.Series(dtype="float64"),
        "net_ret": pd.Series(dtype="float64"), "n_selected": pd.Series(dtype="int64"),
        "n_eligible": pd.Series(dtype="int64")})[list(TRADES_COLUMNS)]


def _simulate(panel: _Panel, sel_cols: Sequence, *, x: float, core: Optional[np.ndarray],
              financing: Optional[np.ndarray], control: bool = False) -> dict:
    """Run the account through every window of ``panel``. ``sel_cols`` gives
    each window's cohort columns (real or placebo). ``control=True`` runs the
    zero-signal equal-weight whole account instead (no SPY core, no financing,
    no short leg): the all-eligible basket held at full equity."""
    n = panel.n
    daily = np.zeros(n)
    overlay_frac = np.zeros(n)
    mask = np.zeros(n, dtype=bool)
    rows = []
    equity = 1.0
    start_at = {panel.entry_pos[k]: k for k in range(len(panel.windows))}
    days_per_year = float(account_relative.CALENDAR_DAYS_PER_YEAR)
    current = None
    for t in range(n):
        e_prev = equity
        if not np.isfinite(e_prev) or e_prev <= 0.0:
            raise ValueError(f"account equity is {e_prev} before {panel.idx[t].date()}; "
                             "the overlay wiped the account")
        k = start_at.get(t)
        if k is not None:
            unit_all, cost_all = panel.all_leg(k)
            if control:
                unit_sel, cost_sel = np.zeros_like(unit_all), 0.0
            else:
                unit_sel = panel.unit_pnl(k, sel_cols[k])
                cost_sel = float(panel.cost[sel_cols[k]].mean())
            current = {"k": k, "size": e_prev, "unit_sel": unit_sel, "unit_all": unit_all,
                       "cost": cost_sel + cost_all if not control else cost_all,
                       "price": 0.0, "fin": 0.0}
        overlay = 0.0
        if current is not None:
            j = t - panel.entry_pos[current["k"]]
            size = current["size"]
            if control:
                price = size * current["unit_all"][j]
                fin = 0.0
            else:
                price = x * size * (current["unit_sel"][j] - current["unit_all"][j])
                rate = financing[t]
                if not np.isfinite(rate):
                    raise ValueError(
                        f"financing rate is not finite on {panel.idx[t].date()}, a session the "
                        "overlay is financed; refusing to assume a zero borrowing cost")
                days = 1.0 if j == 0 else panel.elapsed[t]
                fin = x * size * rate * days / days_per_year
            overlay = price - fin
            current["price"] += price
            current["fin"] += fin
            mask[t] = True
            if t == panel.exit_pos[current["k"]]:
                cost = size * x * current["cost"]          # control runs with x == 1.0
                overlay -= cost
                rows.append({
                    "entry_date": panel.idx[panel.entry_pos[current["k"]]],
                    "exit_date": panel.idx[t],
                    "held": int(t - panel.entry_pos[current["k"]] + 1),
                    "reason": "window", "size": size,
                    "gross_ret": current["price"] / size,
                    "financing_ret": current["fin"] / size,
                    "cost_ret": cost / size,
                    "net_ret": (current["price"] - current["fin"] - cost) / size,
                    "n_selected": int(len(panel.all_cols[current["k"]]) if control
                                      else len(sel_cols[current["k"]])),
                    "n_eligible": int(len(panel.all_cols[current["k"]])),
                })
                current = None
        if control:
            value = overlay / e_prev
        else:
            value = core[t] + overlay / e_prev
        if not np.isfinite(value) or value <= -1.0:
            raise ValueError(f"account return {value} at {panel.idx[t].date()} is not a valid simple return")
        daily[t] = value
        overlay_frac[t] = overlay / e_prev
        equity = e_prev * (1.0 + value)
    return {"daily": daily, "overlay_frac": overlay_frac, "mask": mask, "trades": _trades_frame(rows)}


def _core_returns(panel: _Panel, spy_entry_cost_bps) -> tuple:
    entry_cost = _validated_scalar(spy_entry_cost_bps, "spy_entry_cost_bps", low=0.0, high=1e4,
                                   inclusive_low=True)
    series = cost_adjusted_benchmark_daily_returns(panel.spy, entry_cost_bps=entry_cost).reindex(panel.idx)
    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("benchmark core returns must be finite on every account session")
    return series, values


def _financing_array(panel: _Panel, financing_rate_annual) -> np.ndarray:
    return pd.Series(financing_rate_annual, dtype=float).reindex(panel.idx).to_numpy(dtype=float)


def _real_selection(panel: _Panel) -> list:
    return list(panel.sel_cols)


def build_cohort_overlay_curve(spy_frame, member_frames: Mapping, windows: Sequence, *,
                               cost_bps_by_symbol: Mapping, spy_entry_cost_bps: float,
                               financing_rate_annual, overlay_notional: float) -> dict:
    """The SCORED account curve: SPY held at all times plus, inside each
    window, ``overlay_notional`` times (equal-weight selected cohort minus the
    equal-weight basket of all eligible names). The short basket is a RESEARCH
    CONTROL, not an investable leg (see the module docstring).

    Returns the ``{"strategy_daily", "in_market", "trades_df"}`` dict
    ``abi1_driver.evaluate_family`` accepts from any builder, declaring
    ``account_kind == "benchmark_plus_overlay"`` (the risk-basis rule), plus
    report-only extras (``core_daily``, ``overlay_daily``, ``overlay_mask``,
    ``financing_rate``). ``spy_frame`` is the benchmark's history (a superset
    of the account calendar); ``spy_entry_cost_bps`` is the benchmark's own
    already-halved entry cost, the identical figure the ABI-1 evaluator
    charges, so the core equals the benchmark series exactly."""
    x = _validated_scalar(overlay_notional, "overlay_notional", low=0.0,
                          high=MAX_COHORT_OVERLAY_NOTIONAL, inclusive_low=False)
    panel = _Panel(spy_frame, member_frames, windows, cost_bps_by_symbol)
    core_series, core = _core_returns(panel, spy_entry_cost_bps)
    financing = _financing_array(panel, financing_rate_annual)
    run = _simulate(panel, _real_selection(panel), x=x, core=core, financing=financing)
    idx = panel.idx
    return {
        "strategy_daily": pd.Series(run["daily"], index=idx, name="cohort_overlay_account"),
        "in_market": pd.Series(True, index=idx, name="in_market"),
        "trades_df": run["trades"],
        # Risk-basis rule: ABI-1 measures the risk layer on the overlay's own
        # excess over SPY (`abi1_driver.evaluate_family` reads this key).
        "account_kind": account_relative.ACCOUNT_KIND_BENCHMARK_PLUS_OVERLAY,
        "core_daily": core_series.rename("core"),
        "overlay_daily": pd.Series(run["overlay_frac"], index=idx, name="overlay"),
        "overlay_mask": pd.Series(run["mask"], index=idx, name="overlay_mask"),
        "financing_rate": pd.Series(financing, index=idx, name="financing_rate"),
    }


def build_equal_weight_control_curve(spy_frame, member_frames: Mapping, windows: Sequence, *,
                                     cost_bps_by_symbol: Mapping) -> dict:
    """REPORT-ONLY zero-signal control: the whole account holds the
    equal-weight basket of every ELIGIBLE name in each window (the same
    windows, entry Open, exit Close, fixed shares and costs as the overlay's
    short leg, the cohort ignored). It is the RESEARCH CONTROL the overlay
    subtracts, held as an account on its own: scoring it under ABI-1 shows how
    well a panel of the same names scores against SPY with no signal at all.
    No SPY core, no financing, no ``account_kind`` (scored on the whole-account
    risk basis); off-window sessions carry no return and are marked not in the
    market so the evaluator credits its cash convention."""
    panel = _Panel(spy_frame, member_frames, windows, cost_bps_by_symbol)
    run = _simulate(panel, _real_selection(panel), x=1.0, core=None, financing=None, control=True)
    idx = panel.idx
    return {
        "strategy_daily": pd.Series(run["daily"], index=idx, name="equal_weight_control"),
        "in_market": pd.Series(run["mask"], index=idx, name="in_market"),
        "trades_df": run["trades"],
    }


def build_daily_rebalanced_control_curve(spy_frame, member_frames: Mapping, windows: Sequence) -> dict:
    """REPORT-ONLY zero-signal control, GROSS OF COSTS (an upper bound): the
    whole account holds the equal-weight basket of every ELIGIBLE name
    REBALANCED EVERY SESSION, fully invested from the first window's entry to the
    last window's exit (each session's return is the mean of the eligible names'
    close-to-close returns; the eligible set is that of the window covering the
    session; the cohort is ignored). It is the closest construction to 'the
    equal-weight of every name with enough history' and shows how much of a
    panel's excess over SPY a survivorship-biased, daily-rebalanced basket earns
    with no signal at all. A RESEARCH CONTROL, not an investable account: daily
    rebalancing of hundreds of names is not costed here. Fails closed on any
    eligible name lacking the decision-session close (the first session's
    reference) or a window Close. No ``account_kind`` (whole-account basis)."""
    if not windows:
        raise ValueError("windows must contain at least one CohortWindow")
    zero_costs = {t: 0.0 for w in windows for t in w.eligible}
    panel = _Panel(spy_frame, member_frames, windows, zero_costs)
    daily = np.zeros(panel.n)
    mask = np.zeros(panel.n, dtype=bool)
    rows = []
    for k in range(len(panel.windows)):
        cols = panel.all_cols[k]
        e, x = panel.entry_pos[k], panel.exit_pos[k]
        reference = panel.close[e - 1, cols]
        if np.isnan(reference).any():
            missing = [panel.names[c] for c in cols[np.flatnonzero(np.isnan(reference))]]
            raise CohortDataError(
                "missing_member_price",
                f"window {k}: {len(missing)} eligible name(s) lack the decision-session Close "
                f"{panel.idx[e - 1].date()}: {', '.join(missing[:10])}; refusing to drop or fill them",
                tickers=missing, dates=[panel.idx[e - 1]])
        closes = panel.close[e - 1:x + 1][:, cols]
        returns = (closes[1:] / closes[:-1] - 1.0).mean(axis=1)
        daily[e:x + 1] = returns
        mask[e:x + 1] = True
        net = float(np.prod(1.0 + returns) - 1.0)
        rows.append({"entry_date": panel.idx[e], "exit_date": panel.idx[x], "held": int(x - e + 1),
                     "reason": "window", "size": 1.0, "gross_ret": net, "financing_ret": 0.0,
                     "cost_ret": 0.0, "net_ret": net, "n_selected": int(len(cols)),
                     "n_eligible": int(len(cols))})
    return {
        "strategy_daily": pd.Series(daily, index=panel.idx, name="equal_weight_daily_rebalanced_control"),
        "in_market": pd.Series(mask, index=panel.idx, name="in_market"),
        "trades_df": _trades_frame(rows),
    }


# ----------------------------------------------------------- account identity

# Arithmetic residue, not a return: the account is built as core + overlay (one float addition per session),
# so the recomputed difference is at most a few ulps of the daily return. Far under the smallest real
# overlay session move; a fail-closed guard's tolerance only, it never enters a score.
ACCOUNT_IDENTITY_RESIDUE = 1e-12


def require_account_is_benchmark_plus_overlay(curve: dict, benchmark_daily, *, family_id: str,
                                              tolerance: float = ACCOUNT_IDENTITY_RESIDUE) -> None:
    """Raise unless, on EVERY session of the account (inside the windows and outside them), the curve's
    ``strategy_daily`` equals ``benchmark_daily + overlay_daily`` and its ``core_daily`` equals
    ``benchmark_daily``, to arithmetic residue.

    The risk-basis rule scores the overlay-only excess: the account minus the benchmark the evaluator
    subtracts. The shared guard ``overlay_risk_report.require_excess_is_overlay_only`` only checks the
    sessions the overlay is OFF, and a monthly schedule leaves almost none (one, on the real calendar), so
    a fault inside a window (an account that is not the benchmark plus the reported overlay, or a core that
    is not the evaluator's benchmark) would be invisible to it. This recomputes the identity per session
    from the pre-tax curve and the independently computed benchmark, so it does not depend on one session.

    ``ValueError`` when the curve's series are not on the account's own session index;
    ``RuntimeError`` (nothing recorded) when the benchmark has a hole on an account session or the identity
    fails on any session. Never mutates its inputs."""
    account = curve["strategy_daily"]
    for key in ("overlay_daily", "core_daily"):
        if not curve[key].index.equals(account.index):
            raise ValueError(f"{family_id}: {key} must sit on the account's own session index")
    bench = benchmark_daily.reindex(account.index)
    missing = int(bench.isna().sum())
    if missing:
        raise RuntimeError(
            f"{family_id}: the benchmark has {missing} missing session(s) on the account's calendar, so the "
            "account identity cannot be verified. Nothing was recorded.")
    tol = float(tolerance)
    core_gap = (curve["core_daily"] - bench).abs()
    identity_gap = (account - bench - curve["overlay_daily"]).abs()
    bad_core, bad_identity = int((core_gap > tol).sum()), int((identity_gap > tol).sum())
    if bad_identity:
        raise RuntimeError(
            f"the risk-basis rule: {family_id}'s account is not the benchmark plus the overlay on "
            f"{bad_identity} session(s) (largest gap {float(identity_gap.max()):.3g}), so the scored excess "
            "is not the overlay's own. Nothing was recorded.")
    if bad_core:
        raise RuntimeError(
            f"the risk-basis rule: {family_id}'s core differs from the evaluator's benchmark on {bad_core} "
            f"session(s) (largest gap {float(core_gap.max()):.3g}), so the scored excess is not the "
            "overlay's own. Nothing was recorded.")


# -------------------------------------------------------------------- placebo

def overlay_sharpe(curve: dict) -> float:
    """Annualized Sharpe (252 sessions) of the curve's arithmetic overlay
    excess, ``overlay_daily``, over the whole account calendar. NaN when the
    series has no variance."""
    values = curve["overlay_daily"].to_numpy(dtype=float)
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    if not np.isfinite(std) or std <= 0.0:
        return float("nan")
    return float(np.mean(values) / std * np.sqrt(_TRADING_DAYS_PER_YEAR))


def _sharpe_of(values: np.ndarray) -> float:
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    if not np.isfinite(std) or std <= 0.0:
        return float("nan")
    return float(np.mean(values) / std * np.sqrt(_TRADING_DAYS_PER_YEAR))


def permutation_placebo(spy_frame, member_frames: Mapping, windows: Sequence, *,
                        cost_bps_by_symbol: Mapping, spy_entry_cost_bps: float,
                        financing_rate_annual, overlay_notional: float,
                        n_draws: int = 500, seed: int = 0) -> dict:
    """REPORT-ONLY seeded permutation placebo. Each of ``n_draws`` draws
    replaces EVERY window's cohort with a random cohort of the SAME size from
    the SAME eligible set (same windows, same schedule, same costs, financing
    and sizing), rebuilds the account, and records the annualized Sharpe of its
    overlay excess. The long-minus-eligible short leg is a RESEARCH CONTROL,
    not an investable account. LIMIT: every draw picks from the SAME eligible
    set as the real cohort, so the placebo cannot test rank-conditional
    survivorship (a survivorship or index-inclusion effect specific to the
    ranked cohort is present in the real cohort and absent from the draws).

    Returns the draws, summary quantiles, the real rule's own statistic and its
    ``real_percentile`` (share of draws at or below it) and ``p_upper``
    ((1 + draws at or above it) / (n + 1)). Refuses fewer than
    ``MIN_PLACEBO_DRAWS`` draws. Never writes anywhere and never mutates its
    inputs."""
    if isinstance(n_draws, bool) or not isinstance(n_draws, int) or n_draws < MIN_PLACEBO_DRAWS:
        raise ValueError(f"n_draws must be an integer of at least {MIN_PLACEBO_DRAWS}; got {n_draws!r}")
    x = _validated_scalar(overlay_notional, "overlay_notional", low=0.0,
                          high=MAX_COHORT_OVERLAY_NOTIONAL, inclusive_low=False)
    panel = _Panel(spy_frame, member_frames, windows, cost_bps_by_symbol)
    _, core = _core_returns(panel, spy_entry_cost_bps)
    financing = _financing_array(panel, financing_rate_annual)
    real = _simulate(panel, _real_selection(panel), x=x, core=core, financing=financing)
    real_value = _sharpe_of(real["overlay_frac"])
    rng = np.random.default_rng(int(seed))
    draws = []
    for _ in range(int(n_draws)):
        fake_cols = []
        for k in range(len(panel.windows)):
            size = len(panel.sel_cols[k])
            pick = rng.choice(len(panel.all_cols[k]), size=size, replace=False)
            fake_cols.append(panel.all_cols[k][pick])
        run = _simulate(panel, fake_cols, x=x, core=core, financing=financing)
        draws.append(_sharpe_of(run["overlay_frac"]))
    finite = np.array([d for d in draws if np.isfinite(d)], dtype=float)
    if len(finite) == 0 or not np.isfinite(real_value):
        raise ValueError("the placebo statistic is undefined (zero-variance overlay)")
    return {
        "statistic": PLACEBO_STATISTIC,
        "n_draws": int(n_draws), "seed": int(seed), "draws": [float(d) for d in draws],
        "real_value": float(real_value),
        "draws_mean": float(finite.mean()), "draws_std": float(finite.std(ddof=1)),
        "draws_p05": float(np.percentile(finite, 5)), "draws_p50": float(np.percentile(finite, 50)),
        "draws_p95": float(np.percentile(finite, 95)),
        "real_percentile": float((finite <= real_value).mean()),
        "p_upper": float((1 + int((finite >= real_value).sum())) / (len(finite) + 1)),
        "note": "RESEARCH CONTROL: the short leg is the equal-weight basket of all eligible "
                "names and is not an investable account; the draws come from the same eligible "
                "set as the real cohort, so the placebo cannot test rank-conditional survivorship",
    }
