"""Batch6 candidate: synthetic_leveraged_dip_buy_pooled_largecap
(pooled_single_asset).

Buy the dip on a SYNTHETIC 2x margin-financed position in individual
large-cap stocks, pooled across the existing 518-name universe. The design: a synthetic
2x position off each underlying's own full decades-long history (real 2x
single-stock ETFs only began trading 2022-08-09 -- too young for this
pipeline's evidence floor), with decay/cost/spread modeled; explicit
entry/exit rule -20%-from-high / +50%-gain-or-1yr; high-volatility names
are deliberately kept in the pool rather than filtered out.

RULE
----
    dip_threshold = 0.80          # buy when Close <= 80% of the trailing high
    high_lookback = 252           # trading days ("52-week high")
    profit_target_ratio = 1.50    # sell at +50% gain on the SYNTHETIC position
    time_stop_sessions = 252      # or after 1 year, whichever first

    for each T in POOL independently:
      high_t = rolling 252-session max of T's own daily High (inclusive of t);
               undefined until 252 sessions of history exist
      dip_flag_t = Close_t <= 0.80 * high_t
      while FLAT: on the first bar t where dip_flag_t is True, BUY the
        SYNTHETIC 2x position at the OPEN of bar t+1 (entry_price =
        synthetic Open_{t+1})
      while LONG: SELL at the first bar u (u > entry_fill_bar) where EITHER
        synthetic Close_u >= 1.50 * entry_price, OR
        (u - entry_fill_bar) >= 252
        whichever comes first, at synthetic Close_u
      if the dip condition is still True on the bar the position closed on
        (the exit bar itself), a NEW entry is decided there immediately --
        intended ("still oversold, buy again"), not prevented; see "Chained
        re-entries" below.
      one trade = one contiguous holding period; trades of all pool members
        concatenated into one sample, scored as ONE trial.

Why a SYNTHETIC position, not the real (too-young) 2x ETFs: the design
choice. ``src/data_pipeline.py``'s ``build_synthetic_3x`` is already fully
generic in ``leverage`` (leverage is a parameter of its formula, so
leverage=2.0 needs no math change) -- ``build_synthetic_leveraged`` is a
thin sibling carrying single-stock-appropriate defaults (no fund expense
ratio -- there is no fund wrapper, just real shares financed 2x on
standard Reg-T margin;
financing_spread=100bps over the real time-varying ^IRX, vs. the 50bps
index-ETF default, reflecting retail margin-loan economics -- PENDING
VALIDATION, disclosed default, no real 2x single-stock ETF exists to
calibrate against, which is the whole reason this is synthetic).

Trading cost: the underlying stock's own REAL measured spread (from the
same batch5/batch6 quote-capture mechanism every other family uses) --
this position is realized as real margin-financed shares of the actual
stock, not a packaged product, so there is no separate "leveraged
product spread" to model; leverage's true extra cost is financing,
already modeled inside ``build_synthetic_leveraged``.

Engine mapping (mining.backtest_engine.run_backtest / dip_entry_exit_
events, verified against the real engine by an end-to-end agreement test,
not assumed): ``scripts/run_batch5_mining.py``'s own driver calls
``run_backtest`` WITHOUT a ``time_stop`` argument for the
``pooled_single_asset`` shape (see ``run_single_asset_windows``), so BOTH
the price-target exit and
the time-stop exit must be precomputed by this module itself --
``dip_entry_exit_events`` does that as an isolated, unit-testable pure
function (mirroring ``large_cap_single_stock_pooled_tsmom.decision_
dates()``'s own precompute-once pattern), not the engine's built-in
``time_stop`` mechanism.

Chained re-entries (disclosed, not architecturally prevented): if the dip
condition remains True on the very bar a prior position closed on (most
plausible after a TIME-STOP exit on a persistently-declining name; the
entry/exit price relationship makes it far less likely right after a
PROFIT-TARGET exit, since price is then at or above the old high), the
walk decides a new entry there immediately, filling the next bar --
"still oversold, buy again". This could slice one long decline into
several correlated ~1-year trades; it is not prevented by a cooldown here
(a cooldown would be a CHANGE to the catalog rule, not a bug fix).

Interpretation choices
-----------------------
* Trailing high: the underlying's own real ``High`` column, rolling
  252-session max, ``min_periods=252`` (full window required -- no
  decision before). Matches ``mining/signals.py``'s
  ``fifty_two_week_high_momentum_signal_fn``, this repo's one existing
  "52-week high" precedent (``LOOKBACK_DAYS=252``, ``window["High"].max()``)
  -- NOT that module's own partial-window fallback, since here the dip
  depth is directly load-bearing on the entry decision (unlike that
  module's decile-band use), so an artificially short window would
  misstate the dip.
* "+50% gain from entry": read as +50% on the SYNTHETIC 2x position's own
  entry price (the tradable instrument, and the natural thing a trader
  watching their account would set a profit target on), NOT +50% on the
  underlying stock's raw price.
* Entry fill price: the SYNTHETIC series' own Open at the fill bar (one
  bar after the dip decision bar), matching ``run_backtest``'s own
  next-open entry convention exactly -- NOT the underlying stock's Open,
  and not the synthetic Close (an approximation that would silently
  diverge from what the real engine actually fills).
* "Auto-adjusted daily close (total return)": the underlying frame is
  ``data_pipeline.fetch_ohlc`` (``auto_adjust=True``); the synthetic
  series is built from it via ``data_pipeline.build_synthetic_leveraged``
  with ``dividend_yield=0.0`` (the underlying's Close is already
  total-return, so no separate dividend add-back is needed or wanted).
* Synthetic OHLC shaping (``synthetic_ohlc_from_close``): a daily-reset
  instrument has no real intraday range, so ``Open_t = Close_{t-1}``
  (``Open_0 = Close_0``), ``High = Low = Close`` -- the economically
  correct convention (real leveraged ETFs also strike NAV once per day at
  the close), not a shortcut. Nothing in ``run_backtest`` ever reads
  High/Low (only Open at entry, Close at exit), so this is safe.
* Pool: the same 518-name ``TRADED`` list as ``large_cap_single_stock_
  pooled_tsmom.py``, copied verbatim -- reuses the existing cache, no
  re-curation, and already contains volatile names (TSLA,
  NVDA, COIN, MSTR all present) -- "don't exclude" is satisfied by
  construction, no separate sub-filter applied.
* A recently-listed name (CRWV, RDDT, ARM, HOOD, ...) simply produces
  zero entry decisions until it accumulates 252 real sessions of its own
  history, then behaves normally -- the same graceful degradation every
  lookback-gated module in this repo already relies on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import data_pipeline
from mining.backtest_engine import SignalResult
from mining.batch5.large_cap_single_stock_pooled_tsmom import TRADED as _TSMOM_TRADED

FAMILY_ID = "synthetic_leveraged_dip_buy_pooled_largecap"
SHAPE = "pooled_single_asset"
SNAPSHOTS: dict = {}
BENCHMARK = "SPY"
PARAMETERS = {
    "dip_threshold_pct": -0.20,
    "entry": "synthetic_open_one_bar_after_dip_decision",
    "exit": "synthetic_close_at_target_or_time_stop",
    "financing_spread_bps": 100,
    "high_convention": "trailing_252_session_high_of_daily_high",
    "high_lookback_days": 252,
    "leverage": 2.0,
    "pool": (
        "the pipeline's cached expanded universe: current S&P 500 + NASDAQ "
        "100 constituents after dedup, with cached OHLC and dated bid/ask "
        "quotes from the pipeline's universe cache. Same resolved pool as "
        "large_cap_single_stock_pooled_tsmom. Survivorship caveat: current "
        "constituents only, same convention as every other pooled family "
        "drawn from this universe."
    ),
    "pooling": "all_pool_trades_concatenated_one_trial",
    "profit_target_pct": 0.50,
    "profit_target_basis": "synthetic_leveraged_entry_price",
    "time_stop_sessions": 252,
    "trading_cost_basis": "underlying_stock_real_quoted_spread",
}
EXPECTED_MIN_TRADES = 1000  # configured, conservative floor (not a measured
# count): a ~500-name pool scanned over decades of daily history should
# produce far more dip entries than this; a run below it signals a data or
# wiring problem rather than a quiet strategy.

DIP_THRESHOLD_RATIO = 1.0 + PARAMETERS["dip_threshold_pct"]           # 0.80
HIGH_LOOKBACK = int(PARAMETERS["high_lookback_days"])                  # 252
PROFIT_TARGET_RATIO = 1.0 + PARAMETERS["profit_target_pct"]            # 1.50
TIME_STOP_SESSIONS = int(PARAMETERS["time_stop_sessions"])             # 252
LEVERAGE = float(PARAMETERS["leverage"])
FINANCING_SPREAD = PARAMETERS["financing_spread_bps"] / 1e4

# TRADED is the SYNTHETIC id per pool member (what the engine actually
# prices/holds); SIGNAL is each one's real underlying ticker (needed only
# for the dip condition, never itself held) -- validate_module_contract
# requires the two to be disjoint, and they are (distinct strings, e.g.
# "TSLA_2x_synthetic" vs "TSLA"). Built from LEVERAGE (not a hardcoded
# "2x") so TRADED can never drift from synthetic_id_for's own naming.
SIGNAL = list(_TSMOM_TRADED)
TRADED = [f"{ticker}_{int(LEVERAGE)}x_synthetic" for ticker in _TSMOM_TRADED]

_ENTRY = SignalResult(entry=True, exit=False)
_EXIT = SignalResult(entry=False, exit=True)
_NONE = SignalResult()


def underlying_of(synthetic_ticker: str) -> str:
    """The real ticker a synthetic id (e.g. 'TSLA_2x_synthetic') derives
    from. Inverse of the naming build_synthetic_3x/build_synthetic_
    leveraged already use (f'{ticker}_{int(leverage)}x_synthetic')."""
    suffix = f"_{int(LEVERAGE)}x_synthetic"
    if not synthetic_ticker.endswith(suffix):
        raise ValueError(
            f"{synthetic_ticker!r} is not a {suffix!r}-suffixed synthetic id")
    return synthetic_ticker[: -len(suffix)]


def synthetic_id_for(underlying_ticker: str) -> str:
    return f"{underlying_ticker}_{int(LEVERAGE)}x_synthetic"


def synthetic_ohlc_from_close(dates, close, *, name: str) -> pd.DataFrame:
    """A daily-reset synthetic instrument has no real intraday range:
    ``Open_t = Close_{t-1}`` (``Open_0 = Close_0``), ``High = Low =
    Close``. This is the economically correct convention for an
    instrument that (like a real leveraged ETF) marks its NAV once per
    day at the close, not a shortcut -- and ``run_backtest`` never reads
    High/Low (only Open at entry, Close at exit), so no information is
    lost to downstream consumers. Returns a frame indexed by ``dates``
    with columns Open/High/Low/Close, ready for
    ``run_batch3_mining._normalize_ohlc`` (finite+positive only, no
    High>=Low consistency required)."""
    close = np.asarray(close, dtype=float)
    if len(close) == 0:
        return pd.DataFrame(
            {"Open": [], "High": [], "Low": [], "Close": []},
            index=pd.DatetimeIndex([], name="Date"))
    open_ = np.concatenate([[close[0]], close[:-1]])
    index = pd.DatetimeIndex(pd.to_datetime(dates), name="Date")
    return pd.DataFrame(
        {"Open": open_, "High": close, "Low": close, "Close": close},
        index=index)


def dip_entry_exit_events(underlying_close, underlying_high,
                          synthetic_open, synthetic_close, *,
                          dip_threshold_ratio: float = DIP_THRESHOLD_RATIO,
                          high_lookback: int = HIGH_LOOKBACK,
                          profit_target_ratio: float = PROFIT_TARGET_RATIO,
                          time_stop_sessions: int = TIME_STOP_SESSIONS):
    """Pure sequential walk over one ticker's full (or truncated) shared
    calendar (``underlying_close``/``underlying_high``/``synthetic_open``/
    ``synthetic_close`` must all be the same length, one entry per bar).
    Returns ``(entry_decision_bars, exit_bars)`` -- sorted int arrays of
    0-indexed bar POSITIONS, each entry paired 1:1 with the exit at the
    same array position.

    Bar accounting matches ``backtest_engine.run_backtest`` EXACTLY
    (verified by an end-to-end agreement test, not assumed from reading
    the engine alone -- see the ``test_end_to_end_*`` tests in
    ``test_batch5_synthetic_leveraged_dip_buy_pooled_largecap.py``):
    ``entry_decision_bars[k] = t`` means the dip condition is True at bar
    t while FLAT (``signal_fn`` returns entry=True when queried with
    decision index t); the engine fills at bar ``t + 1``'s open, so this
    walk uses ``synthetic_open[t + 1]`` as the real fill price for the
    profit-target check -- never an approximation from ``synthetic_close``.
    ``exit_bars[k] = u`` means the position closes (target hit OR held ==
    time_stop_sessions) at bar u, the SAME bar ``signal_fn`` returns
    exit=True at (matching the engine's same-bar exit-decide-and-fill
    convention). The earliest u the walk ever considers is
    ``t + 1 + 1 = t + 2`` -- the fill bar itself is consumed by the
    engine's own position-open iteration (an unconditional ``continue``
    with no exit check), confirmed against the real engine.

    After an exit at bar u, the walk resumes checking the dip condition
    starting AT bar u itself (not u + 1) -- the engine's own FLAT branch
    on the iteration immediately after a close is queried with decision
    index u (data through the exit bar), so a dip flag still True there
    creates a new entry decided at u, filling at u + 1. This is the
    "chained re-entry" the family module's own docstring discloses.

    A dip flagged too late for its entry to fill (t + 1 out of range), or
    a position still open at the end of the input, is dropped silently --
    no entry without a matching exit is ever returned.

    Disclosed limitation, not fixed:
    both the profit-target exit and the time-stop exit are surfaced to
    the engine as an ordinary ``exit=True`` signal, so
    ``backtest_engine.run_backtest``'s own ``trades_df["reason"]`` always
    reads ``"signal_exit"`` for this family, never ``"time_stop"`` --
    the driver never passes an engine-level ``time_stop`` (matching this
    module's own documented reason why, above). ``net_ret``/DSR scoring
    is unaffected; a future reader wanting to measure how often the
    252-session cap bound (vs. the profit target) needs to re-derive it
    from ``held``, not from ``reason``."""
    n = len(underlying_close)
    if not (len(underlying_high) == len(synthetic_open) == len(synthetic_close) == n):
        raise ValueError(
            "underlying_close/underlying_high/synthetic_open/synthetic_close "
            "must all share one calendar (equal length)")
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    close = np.asarray(underlying_close, dtype=float)
    high = np.asarray(underlying_high, dtype=float)
    syn_open = np.asarray(synthetic_open, dtype=float)
    syn_close = np.asarray(synthetic_close, dtype=float)

    high_roll = pd.Series(high).rolling(
        high_lookback, min_periods=high_lookback).max().to_numpy()
    with np.errstate(invalid="ignore"):
        dip_flag = np.where(
            np.isnan(high_roll), False, close <= dip_threshold_ratio * high_roll)

    entries: list = []
    exits: list = []
    i = 0
    while i < n:
        if not dip_flag[i]:
            i += 1
            continue
        decision_bar = i
        fill_bar = i + 1
        if fill_bar >= n:
            break  # no bar left to fill this entry -- can't open
        entry_price = float(syn_open[fill_bar])

        exit_bar = None
        j = fill_bar + 1  # earliest bar the engine can check for exit
        while j < n:
            held = j - fill_bar
            if syn_close[j] >= profit_target_ratio * entry_price or held >= time_stop_sessions:
                exit_bar = j
                break
            j += 1
        if exit_bar is None:
            break  # position never closes within this truncated history

        entries.append(decision_bar)
        exits.append(exit_bar)
        i = exit_bar  # resume the FLAT scan AT the exit bar (matches the
                      # engine's own decision-index convention on the
                      # iteration immediately after a close)

    return np.array(entries, dtype=np.int64), np.array(exits, dtype=np.int64)


def make_signal_fn(ticker: str, index, frames: dict):
    """SignalFn for one pool member's SYNTHETIC id over one bucket
    (``index``). ``ticker`` is the synthetic id (e.g. 'TSLA_2x_synthetic');
    ``frames`` must contain BOTH ``frames[ticker]`` (the synthetic OHLC,
    built by the driver) and ``frames[underlying_of(ticker)]`` (the real
    underlying OHLC, needed for the dip signal) -- both truncated to
    ``<= index[-1]`` before any computation, holdout-blind by construction
    (same pattern as ``large_cap_single_stock_pooled_tsmom.make_signal_
    fn``: pre-bucket history is used only as lookback warm-up)."""
    underlying = underlying_of(ticker)
    if ticker not in frames:
        raise KeyError(f"{FAMILY_ID}: no synthetic frame for {ticker}")
    if underlying not in frames:
        raise KeyError(f"{FAMILY_ID}: no underlying frame for {underlying}")
    syn_frame = frames[ticker]
    und_frame = frames[underlying]
    for name, frame in ((ticker, syn_frame), (underlying, und_frame)):
        if not {"Open", "High", "Close"}.issubset(frame.columns):
            raise ValueError(f"{FAMILY_ID}: {name} frame lacks Open/High/Close")

    bucket = pd.DatetimeIndex(pd.to_datetime(index))
    if bucket.tz is not None:
        bucket = bucket.tz_localize(None)
    if len(bucket) == 0:
        def empty_signal_fn(frame_slice, i):
            return _NONE
        return empty_signal_fn
    cutoff = bucket[-1]

    def _prep(frame):
        idx = pd.DatetimeIndex(pd.to_datetime(frame.index))
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        out = frame.set_axis(idx)
        return out.loc[out.index <= cutoff]

    syn_frame = _prep(syn_frame)
    und_frame = _prep(und_frame)
    if not syn_frame.index.equals(und_frame.index):
        raise ValueError(
            f"{FAMILY_ID}: {ticker} synthetic and {underlying} underlying "
            "frames must share one identical calendar up to the bucket cutoff")

    entries, exit_bars = dip_entry_exit_events(
        und_frame["Close"].to_numpy(dtype=float),
        und_frame["High"].to_numpy(dtype=float),
        syn_frame["Open"].to_numpy(dtype=float),
        syn_frame["Close"].to_numpy(dtype=float))

    full_dates = syn_frame.index
    entry_dates = full_dates[entries] if len(entries) else pd.DatetimeIndex([])
    exit_dates = full_dates[exit_bars] if len(exit_bars) else pd.DatetimeIndex([])
    bucket_entry = np.asarray(bucket.isin(entry_dates), dtype=bool)
    bucket_exit = np.asarray(bucket.isin(exit_dates), dtype=bool)
    results = np.empty(len(bucket), dtype=object)
    for pos in range(len(bucket)):
        if bucket_entry[pos] and bucket_exit[pos]:
            results[pos] = SignalResult(entry=True, exit=True)
        elif bucket_entry[pos]:
            results[pos] = _ENTRY
        elif bucket_exit[pos]:
            results[pos] = _EXIT
        else:
            results[pos] = _NONE

    bucket_ns = bucket.asi8
    by_date = {int(ts): pos for pos, ts in enumerate(bucket_ns)}
    n = len(bucket)

    def signal_fn(frame_slice, i):
        today = pd.Timestamp(frame_slice.index[-1])
        if today.tzinfo is not None:
            today = today.tz_localize(None)
        today_ns = int(today.normalize().value)
        if 0 <= i < n and bucket_ns[i] == today_ns:
            return results[i]
        pos = by_date.get(today_ns)
        if pos is None:
            return _NONE
        return results[pos]

    return signal_fn
