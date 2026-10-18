"""Generic daily-bar backtest engine for the strategy mining pipeline.

Deliberately NOT a fixed-rule engine that hardcodes RSI2/IBS/MA5 logic.
This engine takes an arbitrary entry/exit SIGNAL FUNCTION instead: `run_backtest(df, signal_fn, ...) ->
(trades_df, in_market_series)`.

Self-contained: it depends only on numpy and pandas, not on any other
backtest or gate module.

This is a ROUGH SCREEN, not a real-capital execution engine: it does not
model every order-collision nuance of a production execution rule (e.g.
same-day entry+exit evaluation is not attempted -- exit checks begin the
bar after entry). Any later use of a surviving candidate needs a precise
execution model of its own; this engine's job is only to rank candidates
fairly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalResult:
    """One bar's signal state, returned by a candidate's signal function."""
    entry: bool = False
    exit: bool = False


SignalFn = Callable[[pd.DataFrame, int], "SignalResult"]
"""A signal function receives a SLICE of the OHLC DataFrame ending
EXACTLY at the bar it may look at, plus that bar's position `i` in the
ORIGINAL (unsliced) frame. `frame` always has exactly `i + 1` rows, so
`frame.iloc[-1]` (equivalently `frame.iloc[i]`) is "today" and there is
no row for "tomorrow" to accidentally read -- `frame.iloc[i + 1]` raises
IndexError. This makes look-ahead bias STRUCTURALLY impossible rather
than merely forbidden by convention, which matters because the signal functions may be generated from prose
descriptions at run time.

The engine acts on an entry signal at the NEXT bar's open (next-open
execution -- the only style that is unambiguously executable) and on an exit signal at the
SAME bar's close (the pinned exit rule). Entry and exit are therefore
priced at different points of the bar; that timing asymmetry is part of
the rough-screen model and applies equally to every candidate."""


def run_backtest(df: pd.DataFrame, signal_fn: SignalFn, *,
                  cost_bps: float = 0.0, tax_haircut_rate: float = 0.0,
                  time_stop: Optional[int] = None,
                  direction: Literal["long", "short"] = "long"):
    """Run one candidate's signal function over `df` (columns Open, High,
    Low, Close; index ascending trading-day timestamps).

    `direction` is "long" (the default) or "short". Short gross return is simple one-share P&L on entry capital:
    `1 - exit_px / entry_px`. `cost_bps` must be a ROUND-TRIP figure (e.g. from
    cost_tax_inputs.session_round_trip) -- a one-way spread silently
    halves the modeled friction. `tax_haircut_rate` must be in [0, 1);
    raises ValueError otherwise.

    Returns (trades_df, in_market_series):
    - trades_df columns: entry_date, exit_date, entry_px, exit_px, held,
      reason ("signal_exit" | "time_stop" | "end_of_data" -- when both a
      time-stop and a signal-exit fire on the same bar, "signal_exit"
      wins; a downstream count of "how often the time stop bound" will
      therefore undercount by exactly that overlap), gross_ret, net_ret
      (after cost_bps and tax_haircut_rate; the haircut applies only when
      net_ret is positive). Always has correct per-column dtypes (float
      for prices/returns, int for held) even with zero trades -- never
      falls back to object dtype, so a downstream scipy.stats call on an
      empty result behaves the same as the non-empty case.
    - in_market_series: bool Series, same index as df, True from the bar
      a position is OPENED through the bar it is CLOSED on, inclusive.

    Degenerate edge case, disclosed not fixed: an entry on the FINAL bar
    of `df` is force-closed the same bar via the end_of_data path with
    held=0 -- reachable only with very short input data.
    """
    if not (0.0 <= tax_haircut_rate < 1.0):
        raise ValueError(f"tax_haircut_rate must be in [0, 1), got {tax_haircut_rate}")
    if direction not in ("long", "short"):
        raise ValueError(
            f"direction must be 'long' or 'short', got {direction!r}")

    n = len(df)
    in_market = np.zeros(n, dtype=bool)
    trades = []
    position = None

    def _close_trade(i, reason):
        exit_px = float(df["Close"].iloc[i])
        long_ret = exit_px / position["entry_px"] - 1.0
        gross_ret = long_ret if direction == "long" else -long_ret
        net_ret = gross_ret - cost_bps / 1e4
        if tax_haircut_rate and net_ret > 0:
            net_ret *= (1.0 - tax_haircut_rate)
        trades.append({
            "entry_date": df.index[position["entry_idx"]],
            "exit_date": df.index[i],
            "entry_px": position["entry_px"],
            "exit_px": exit_px,
            "held": i - position["entry_idx"],
            "reason": reason,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
        })

    for i in range(n):
        if position is None:
            if i > 0:
                sig = signal_fn(df.iloc[:i], i - 1)
                if sig.entry:
                    position = {"entry_idx": i, "entry_px": float(df["Open"].iloc[i])}
                    in_market[i] = True
            continue

        in_market[i] = True
        held = i - position["entry_idx"]
        sig = signal_fn(df.iloc[:i + 1], i)
        hit_time_stop = time_stop is not None and held >= time_stop
        if sig.exit or hit_time_stop:
            _close_trade(i, "time_stop" if hit_time_stop and not sig.exit else "signal_exit")
            position = None

    if position is not None:
        _close_trade(n - 1, "end_of_data")

    if trades:
        trades_df = pd.DataFrame(trades, columns=[
            "entry_date", "exit_date", "entry_px", "exit_px", "held",
            "reason", "gross_ret", "net_ret"])
    else:
        trades_df = pd.DataFrame({
            "entry_date": pd.Series(dtype="datetime64[ns]"),
            "exit_date": pd.Series(dtype="datetime64[ns]"),
            "entry_px": pd.Series(dtype="float64"),
            "exit_px": pd.Series(dtype="float64"),
            "held": pd.Series(dtype="int64"),
            "reason": pd.Series(dtype="object"),
            "gross_ret": pd.Series(dtype="float64"),
            "net_ret": pd.Series(dtype="float64"),
        })
    return trades_df, pd.Series(in_market, index=df.index, name="in_market")


def holdout_passes(in_sample_net_edge_bps: float,
                    holdout_net_edge_bps: float) -> bool:
    """Per-trade holdout pass rule: directional consistency, not an
    equally strict statistical bar -- 6 months is too small a sample for
    a matching significance test. Passes iff the holdout edge is POSITIVE
    and the in-sample edge that earned the holdout look was also
    positive (in practice always true at the holdout stage, since the
    preceding DSR bar already filters on a positive edge)."""
    if in_sample_net_edge_bps <= 0:
        return False
    return holdout_net_edge_bps > 0


def daily_returns_by_exit_day(index: pd.DatetimeIndex,
                               trades_df: pd.DataFrame) -> pd.Series:
    """A daily return series aligned to `index`, zero everywhere except
    each trade's exit date, which carries that trade's full net_ret as a
    lump-sum realization. Used ONLY for the DSR's cross-candidate
    correlation estimate (dsr.num_independent_trials, Bailey & Lopez de
    Prado 2014 Appendix A.3) -- a common-calendar proxy so different
    candidates' trade streams can be correlated at all, not a claim about
    intraday P&L attribution.

    Raises ValueError if any trade's exit_date is not in `index` --
    fail closed rather than silently dropping a trade, since a dropped
    trade would silently degrade the correlation estimate with no error
    anywhere downstream. This means `index` must be `trades_df`'s OWN
    source dataframe's index, not an arbitrary or differently-windowed
    one."""
    s = pd.Series(0.0, index=index)
    for _, row in trades_df.iterrows():
        if row["exit_date"] not in s.index:
            raise ValueError(
                f"trade exit_date {row['exit_date']} is not in the provided "
                f"index -- pass the same df.index that produced trades_df")
        s.loc[row["exit_date"]] += row["net_ret"]
    return s
