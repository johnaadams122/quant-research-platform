"""Reusable SignalFn and regime/target implementations for catalog
candidates.

The single-ticker candidates here (52-week-high momentum, post-earnings
drift, whose signal comes from one ticker's own price/earnings history)
take no hardcoded ticker: the same function runs unmodified against every
ticker's own OHLC frame (PEAD's per-ticker earnings dates arrive through
its factory), so running the SAME mechanism
across a broad universe (e.g. S&P 500 + NASDAQ 100, ~500 tickers) is a
larger-sample test of the same family, not a redefinition of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest_engine import SignalResult

# ---------------------------------------------------------------------------
# fifty_two_week_high_momentum (catalog parameters: lookback_days=252,
# rank_decile='top', rebalance='monthly'). Takes no per-ticker state -- the
# same function object runs unmodified against every ticker's own OHLC
# frame, which is the whole basis for pooling its trades across tickers as
# one larger sample of the SAME mechanism.
# ---------------------------------------------------------------------------

LOOKBACK_DAYS = 252  # candidate['parameters']['lookback_days']
TOP_DECILE_FRACTION = 0.10  # "closest-to-high decile" proxy: price within
# the top 10% of the trailing lookback window's [low, high] range


def fifty_two_week_high_momentum_signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
    if len(frame) < 2:
        return SignalResult()

    today = frame.index[-1]
    prev = frame.index[-2]
    is_rebalance_day = (today.year, today.month) != (prev.year, prev.month)
    if not is_rebalance_day:
        return SignalResult()

    window = frame.iloc[-LOOKBACK_DAYS:] if len(frame) >= LOOKBACK_DAYS else frame
    high = float(window["High"].max())
    low = float(window["Low"].min())
    price = float(frame["Close"].iloc[-1])
    band = high - low
    if band <= 0:
        return SignalResult(entry=False, exit=True)

    threshold = high - TOP_DECILE_FRACTION * band
    want_long = price >= threshold
    return SignalResult(entry=want_long, exit=not want_long)


# ---------------------------------------------------------------------------
# pead_earnings_surprise_drift (catalog parameters: hold_days=60,
# entry_trigger='earnings_day_return > +1 stdev(trailing_60d_daily_
# returns)'). Per-ticker state (its real earnings-announcement dates) is
# supplied via a factory rather than a module-level constant, since -- unlike
# the momentum signal above -- this mechanism is genuinely per-ticker: each
# stock has its own earnings calendar.
# ---------------------------------------------------------------------------

TRAILING_WINDOW_DAYS = 60


def make_pead_signal_fn(earnings_dates: frozenset, trailing_window_days: int = TRAILING_WINDOW_DAYS):
    """Returns a SignalFn closed over `earnings_dates` (a frozenset of
    'YYYY-MM-DD' strings -- that ticker's own real historical
    earnings-announcement dates, restricted to in-sample or holdout dates
    by the caller as appropriate). Entry logic: on a real earnings date, if
    that day's close-to-close return exceeds +1 standard deviation of the
    trailing `trailing_window_days` daily returns (excluding the
    announcement day's own return), enter long; the engine's own
    next-open execution handles "enter at the next session's open". Never
    sets exit=True -- exit is via the engine's time_stop=hold_days
    mechanism only, matching the candidate's mechanism."""

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        if len(frame) < trailing_window_days + 2:
            return SignalResult()

        today = frame.index[-1]
        today_str = today.strftime("%Y-%m-%d")
        if today_str not in earnings_dates:
            return SignalResult()

        closes = frame["Close"]
        if closes.iloc[-2] <= 0 or pd.isna(closes.iloc[-1]) or pd.isna(closes.iloc[-2]):
            return SignalResult()

        earnings_day_return = closes.iloc[-1] / closes.iloc[-2] - 1.0

        daily_returns = closes.pct_change()
        trailing = daily_returns.iloc[-(trailing_window_days + 1):-1]
        if len(trailing) < trailing_window_days or trailing.isna().any():
            return SignalResult()

        threshold = trailing.std(ddof=1)
        if pd.isna(threshold):
            return SignalResult()

        if earnings_day_return > threshold:
            return SignalResult(entry=True, exit=False)
        return SignalResult()

    return signal_fn


# ---------------------------------------------------------------------------
# sp500_index_addition_effect (catalog parameters: entry='announcement_date',
# hold_days_post_inclusion=10). Per-ticker state (a real, historically
# verified announcement date and a precomputed exit date) is supplied via a
# factory -- unlike PEAD's recurring per-ticker earnings calendar, an index
# addition happens exactly ONCE in a company's history, so this factory
# takes a single (entry_date, exit_date) pair rather than a set of dates.
# ---------------------------------------------------------------------------

def make_index_addition_signal_fn(entry_date: str, exit_date: str):
    """Returns a SignalFn that enters UNCONDITIONALLY the session after
    `entry_date` (a real, verified S&P 500 addition announcement date --
    next-open execution, same convention as every other signal in this
    module) and exits unconditionally at `exit_date`'s close. `exit_date`
    must be precomputed by the CALLER as the effective inclusion date plus
    hold_days_post_inclusion TRADING days on that specific ticker's own OHLC
    index -- this function has no access to that index and does no date
    arithmetic itself.

    Single-shot by construction: an index addition happens once per ticker,
    so there is no re-entry guard to write -- each real calendar date occurs
    exactly once in any OHLC index, and the engine only evaluates the entry
    branch while no position is open. Precondition (caller's
    responsibility, always true in practice since exit_date is always
    computed forward from a later effective date): entry_date < exit_date.
    If `exit_date` is never reached within the data handed to
    run_backtest, the engine's own end_of_data path force-closes the
    position -- disclosed, not fixed, matching backtest_engine.run_backtest's
    own documented degenerate-edge-case behavior."""

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        today_str = frame.index[-1].strftime("%Y-%m-%d")
        if today_str == exit_date:
            return SignalResult(exit=True)
        if today_str == entry_date:
            return SignalResult(entry=True)
        return SignalResult()

    return signal_fn


def _clean_series(series: pd.Series) -> pd.Series:
    """Return a sorted, unique, finite numeric series."""
    out = pd.Series(series, copy=True)
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).tz_localize(None)
    out = pd.to_numeric(out, errors="coerce")
    nonfinite = out.notna() & ~np.isfinite(out)
    if nonfinite.any():
        dates = [timestamp.strftime("%Y-%m-%d")
                 for timestamp in out.index[nonfinite]]
        raise ValueError(
            f"series contains non-finite values at dates {dates}")
    out = out.dropna()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _attach_source_metadata(callable_object, source: pd.Series):
    """Expose external-series provenance at the callable boundary."""
    metadata = dict(source.attrs)
    callable_object.source_metadata = metadata
    for name in (
            "ledgerable", "ledger_block_reason", "source_url",
            "date_semantics", "point_in_time"):
        if name in metadata:
            setattr(callable_object, name, metadata[name])
    return callable_object


def make_external_regime_signal_fn(risk_on: pd.Series):
    """Create a long/cash SignalFn from a dated boolean risk-on series.

    The lookup is strictly as-of the current traded bar. No backward fill is
    possible, and a bar before the signal's first valid observation is a no-op.
    """
    state = pd.Series(risk_on, copy=True)
    state.index = pd.DatetimeIndex(pd.to_datetime(state.index)).tz_localize(None)
    state = state.dropna()
    state = state[~state.index.duplicated(keep="last")].sort_index().astype(bool)

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        today = pd.Timestamp(frame.index[-1]).tz_localize(None)
        known = state.loc[state.index <= today]
        if known.empty:
            return SignalResult()
        is_risk_on = bool(known.iloc[-1])
        return SignalResult(entry=is_risk_on, exit=not is_risk_on)

    return _attach_source_metadata(signal_fn, risk_on)


def make_date_window_signal_fn(entry_dates, exit_dates):
    """Create a SignalFn from precomputed dates within one bounded window."""
    entries = {pd.Timestamp(d).tz_localize(None).normalize() for d in entry_dates}
    exits = {pd.Timestamp(d).tz_localize(None).normalize() for d in exit_dates}

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        today = pd.Timestamp(frame.index[-1]).tz_localize(None).normalize()
        return SignalResult(entry=today in entries, exit=today in exits)

    return signal_fn


def monthly_opex_dates(
        index: pd.DatetimeIndex, *,
        entry_trading_days_before_opex: int):
    """Signal/exit dates for ordinary third-Friday monthly option expiry.

    ``entry_trading_days_before_opex`` comes directly from the catalog. The
    generic engine acts on entry signals at the *next* session's open, so the
    returned entry-signal date is one further session back than that offset.
    The exit signal is expiry Friday because exits execute at that close. The
    caller must pass the already bucket-bounded traded calendar.
    """
    if (isinstance(entry_trading_days_before_opex, bool) or
            not isinstance(entry_trading_days_before_opex, int) or
            entry_trading_days_before_opex <= 0):
        raise ValueError(
            "entry_trading_days_before_opex must be a positive integer")
    dates = pd.DatetimeIndex(pd.to_datetime(index)).tz_localize(None).sort_values()
    entries, exits = set(), set()
    signal_offset = entry_trading_days_before_opex + 1
    for period in dates.to_period("M").unique():
        month = dates[dates.to_period("M") == period]
        expiry = month[(month.weekday == 4) & (month.day >= 15) & (month.day <= 21)]
        if len(expiry) != 1:
            continue
        exit_date = expiry[0]
        pos = dates.get_loc(exit_date)
        if isinstance(pos, int) and pos >= signal_offset:
            entries.add(dates[pos - signal_offset])
            exits.add(exit_date)
    return frozenset(entries), frozenset(exits)


def post_opex_window_dates(
        index: pd.DatetimeIndex, *,
        hold_trading_days_after_opex: int):
    """Signal/exit dates for the trading window immediately AFTER ordinary
    third-Friday monthly option expiry -- the "post-opex gamma release"
    window, and the mirror image of ``monthly_opex_dates`` (entry BEFORE
    expiry, exit AT expiry). Here the entry signal fires ON expiry Friday
    itself -- the generic engine acts on an entry signal at the *next*
    session's open, so the position opens at the first post-opex session,
    exactly the window this family means to hold. The exit signal fires
    ``hold_trading_days_after_opex`` actual trading sessions after expiry
    Friday (exits execute at that session's close). A month whose expiry
    falls too close to the end of the supplied calendar for the exit date
    to exist within it is skipped entirely -- this function never invents a
    session beyond what ``index`` supplies, matching ``monthly_opex_dates``'s
    own bounds discipline. The caller must pass the already bucket-bounded
    traded calendar.
    """
    if (isinstance(hold_trading_days_after_opex, bool) or
            not isinstance(hold_trading_days_after_opex, int) or
            hold_trading_days_after_opex <= 0):
        raise ValueError(
            "hold_trading_days_after_opex must be a positive integer")
    dates = pd.DatetimeIndex(pd.to_datetime(index)).tz_localize(None).sort_values()
    entries, exits = set(), set()
    for period in dates.to_period("M").unique():
        month = dates[dates.to_period("M") == period]
        expiry = month[(month.weekday == 4) & (month.day >= 15) & (month.day <= 21)]
        if len(expiry) != 1:
            continue
        entry_date = expiry[0]
        pos = dates.get_loc(entry_date)
        if not isinstance(pos, int):
            continue
        exit_pos = pos + hold_trading_days_after_opex
        if exit_pos >= len(dates):
            continue
        entries.add(entry_date)
        exits.add(dates[exit_pos])
    return frozenset(entries), frozenset(exits)


def moving_average_risk_on_series(series: pd.Series, *, fast: int,
                                  slow: int) -> pd.Series:
    """Risk-on when a signal's fast MA is at or below its slow MA."""
    values = _clean_series(series)
    fast_ma = values.rolling(fast, min_periods=fast).mean()
    slow_ma = values.rolling(slow, min_periods=slow).mean()
    valid = fast_ma.notna() & slow_ma.notna()
    return (fast_ma.loc[valid] <= slow_ma.loc[valid]).rename("risk_on")


def ratio_moving_average_risk_on_series(numerator: pd.Series,
                                        denominator: pd.Series, *,
                                        fast: int, slow: int) -> pd.Series:
    """MA regime on a same-date numerator/denominator ratio."""
    aligned = pd.concat(
        [_clean_series(numerator).rename("numerator"),
         _clean_series(denominator).rename("denominator")],
        axis=1, join="inner").dropna()
    aligned = aligned.loc[aligned["denominator"] != 0]
    ratio = aligned["numerator"] / aligned["denominator"]
    return moving_average_risk_on_series(ratio, fast=fast, slow=slow)


def money_market_risk_on_series(series: pd.Series, *, roc_months: int,
                                riskoff_z_threshold: float) -> pd.Series:
    """Causal expanding-z regime for retail money-market fund flows."""
    if roc_months <= 0:
        raise ValueError("roc_months must be positive")
    values = _clean_series(series)
    periods = values.index.to_period("M")
    if periods.duplicated().any():
        raise ValueError("money-market series has duplicate calendar months")
    by_month = pd.Series(values.to_numpy(), index=periods, dtype=float)
    roc_values = {}
    for period, current in by_month.items():
        prior_period = period - roc_months
        if prior_period not in by_month.index or by_month.loc[prior_period] == 0:
            continue
        roc_values[period] = (
            (float(current) - float(by_month.loc[prior_period])) /
            float(by_month.loc[prior_period]))
    roc = pd.Series(
        [roc_values[p] for p in periods if p in roc_values],
        index=[values.index[i] for i, p in enumerate(periods) if p in roc_values],
        dtype=float,
    )
    mean = roc.expanding(min_periods=2).mean()
    std = roc.expanding(min_periods=2).std(ddof=1)
    zscore = (roc - mean) / std
    valid = zscore.notna() & std.gt(0)
    result = (zscore.loc[valid] <= riskoff_z_threshold).rename("risk_on")
    result.attrs.update(series.attrs)
    return result


def fed_net_liquidity_risk_on_series(walcl: pd.Series, wtregen: pd.Series,
                                     rrpontsyd: pd.Series, *,
                                     ma_weeks: int) -> pd.Series:
    """Risk-on when Fed net liquidity is above its trailing weekly MA.

    WALCL and WTREGEN are millions of dollars; RRPONTSYD is billions, so
    the latter is multiplied by 1,000 before subtraction. Lower-frequency
    inputs are forward-filled only onto WALCL's observation dates.
    """
    if ma_weeks <= 0:
        raise ValueError("ma_weeks must be positive")
    assets = _clean_series(walcl)
    tga = _clean_series(wtregen).reindex(assets.index, method="ffill")
    rrp = _clean_series(rrpontsyd).reindex(assets.index, method="ffill")
    net = assets - tga - rrp * 1_000.0
    net = net.dropna()
    week_periods = net.index.to_period("W-SUN")
    if week_periods.duplicated().any():
        raise ValueError("WALCL has duplicate observations in a calendar week")
    by_week = pd.Series(net.to_numpy(), index=week_periods, dtype=float)
    states = {}
    for when, period in zip(net.index, week_periods):
        required = pd.period_range(
            end=period, periods=ma_weeks, freq="W-SUN")
        if not required.isin(by_week.index).all():
            continue
        trailing = by_week.loc[required]
        states[when] = float(by_week.loc[period]) > float(trailing.mean())
    result = pd.Series(states, dtype=bool, name="risk_on")
    if any(source.attrs.get("ledgerable") is False
           for source in (walcl, wtregen, rrpontsyd)):
        result.attrs.update({
            "ledgerable": False,
            "ledger_block_reason": (
                "one or more FRED inputs lack point-in-time publication/"
                "vintage evidence"),
        })
    return result


def _aligned_frames(frames: dict[str, pd.DataFrame],
                    required_tickers: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Validate the pure target-factory contract and return required frames."""
    if not isinstance(frames, dict):
        raise TypeError("target input must be a dict of aligned OHLC frames")
    selected = {}
    common_index = None
    for ticker in required_tickers:
        if ticker not in frames:
            raise ValueError(f"missing required aligned frame for {ticker}")
        frame = frames[ticker]
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"{ticker} frame must be a non-empty DataFrame")
        if "Close" not in frame.columns:
            raise ValueError(f"{ticker} frame is missing Close")
        index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None)
        if not index.is_monotonic_increasing or index.has_duplicates:
            raise ValueError(f"{ticker} frame index must be sorted and unique")
        closes = pd.to_numeric(frame["Close"], errors="coerce")
        if closes.isna().any() or not np.isfinite(closes.to_numpy()).all():
            raise ValueError(f"{ticker} Close values must be finite numeric")
        if common_index is None:
            common_index = index
        elif not index.equals(common_index):
            raise ValueError("rotation target frames must be exactly aligned")
        normalized = frame.copy()
        normalized.index = index
        selected[ticker] = normalized
    return selected


def _monthly_decision_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Prefix ending on the first available session of the current month."""
    periods = frame.index.to_period("M")
    current_period = periods[-1]
    decision_position = int(np.flatnonzero(periods == current_period)[0])
    return frame.iloc[:decision_position + 1]


def _calendar_anchor_close(frame: pd.DataFrame, decision_date: pd.Timestamp,
                           lookback_months: int):
    anchor = decision_date - pd.DateOffset(months=lookback_months)
    eligible = frame.loc[frame.index <= anchor, "Close"]
    if eligible.empty:
        return None
    return float(eligible.iloc[-1])


def make_monthly_absolute_momentum_target_fn(
        risk_ticker: str, defensive_ticker: str, *, lookback_months: int):
    """Monthly absolute-momentum target using a calendar-month anchor."""
    if lookback_months <= 0:
        raise ValueError("lookback_months must be positive")

    def target(frames: dict[str, pd.DataFrame]):
        aligned = _aligned_frames(frames, (risk_ticker, defensive_ticker))
        decision = _monthly_decision_frame(aligned[risk_ticker])
        decision_date = decision.index[-1]
        base = _calendar_anchor_close(
            decision, decision_date, lookback_months)
        if base is None or base <= 0:
            return None
        current = float(decision["Close"].iloc[-1])
        return risk_ticker if current > base else defensive_ticker

    return target


def make_monthly_relative_momentum_target_fn(
        risk_ticker: str, defensive_ticker: str, *, lookback_months: int):
    """Monthly relative-momentum target using the same calendar anchor."""
    if lookback_months <= 0:
        raise ValueError("lookback_months must be positive")

    def target(frames: dict[str, pd.DataFrame]):
        aligned = _aligned_frames(frames, (risk_ticker, defensive_ticker))
        risk = _monthly_decision_frame(aligned[risk_ticker])
        defensive = _monthly_decision_frame(aligned[defensive_ticker])
        decision_date = risk.index[-1]
        risk_base = _calendar_anchor_close(
            risk, decision_date, lookback_months)
        defensive_base = _calendar_anchor_close(
            defensive, decision_date, lookback_months)
        if risk_base is None or defensive_base is None:
            return None
        if risk_base <= 0 or defensive_base <= 0:
            return None
        risk_return = (
            float(risk["Close"].iloc[-1]) - risk_base) / risk_base
        defensive_return = (
            float(defensive["Close"].iloc[-1]) - defensive_base) / defensive_base
        return risk_ticker if risk_return > defensive_return else defensive_ticker

    return target


def make_monthly_volatility_target_fn(
        risk_ticker: str, defensive_ticker: str, *, vol_lookback_days: int,
        median_lookback_days: int = 252):
    """Monthly target from realized vol versus its trailing daily median."""
    if vol_lookback_days < 2 or median_lookback_days <= 0:
        raise ValueError("volatility lookbacks must be positive and vol >= 2")

    def target(frames: dict[str, pd.DataFrame]):
        aligned = _aligned_frames(frames, (risk_ticker, defensive_ticker))
        decision = _monthly_decision_frame(aligned[risk_ticker])
        returns = decision["Close"].pct_change(fill_method=None).dropna()
        realized = (returns.rolling(
            vol_lookback_days, min_periods=vol_lookback_days).std(ddof=1) *
            np.sqrt(252.0)).dropna()
        if len(realized) < median_lookback_days:
            return None
        current = float(realized.iloc[-1])
        median = float(realized.iloc[-median_lookback_days:].median())
        return risk_ticker if current < median else defensive_ticker

    return target


def make_moving_average_regime_target_fn(
        *, risk_ticker: str, defensive_ticker: str, signal_ticker: str,
        fast: int, slow: int):
    """Stable daily target for SKEW-like fast/slow MA regimes."""
    if fast <= 0 or slow <= 0 or fast >= slow:
        raise ValueError("moving-average windows require 0 < fast < slow")

    def target(frames: dict[str, pd.DataFrame]):
        aligned = _aligned_frames(
            frames, (risk_ticker, defensive_ticker, signal_ticker))
        values = aligned[signal_ticker]["Close"]
        if len(values) < slow:
            return None
        fast_ma = float(values.iloc[-fast:].mean())
        slow_ma = float(values.iloc[-slow:].mean())
        return risk_ticker if fast_ma <= slow_ma else defensive_ticker

    return target


def make_ratio_regime_target_fn(
        *, risk_ticker: str, defensive_ticker: str, numerator_ticker: str,
        denominator_ticker: str, fast: int, slow: int):
    """Stable daily target for a fast/slow MA of an aligned ratio."""
    if fast <= 0 or slow <= 0 or fast >= slow:
        raise ValueError("moving-average windows require 0 < fast < slow")

    def target(frames: dict[str, pd.DataFrame]):
        aligned = _aligned_frames(frames, (
            risk_ticker, defensive_ticker, numerator_ticker,
            denominator_ticker))
        denominator = aligned[denominator_ticker]["Close"]
        if (denominator == 0).any():
            raise ValueError("ratio denominator contains zero")
        ratio = aligned[numerator_ticker]["Close"] / denominator
        if len(ratio) < slow:
            return None
        fast_ma = float(ratio.iloc[-fast:].mean())
        slow_ma = float(ratio.iloc[-slow:].mean())
        return risk_ticker if fast_ma <= slow_ma else defensive_ticker

    return target


def make_external_regime_target_fn(
        risk_on: pd.Series, *, risk_ticker: str,
        defensive_ticker: str | None):
    """As-of external regime target; dates before first release stay flat."""
    state = pd.Series(risk_on, copy=True)
    state.index = pd.DatetimeIndex(pd.to_datetime(state.index)).tz_localize(None)
    state = state.dropna()
    state = state[~state.index.duplicated(keep="last")].sort_index().astype(bool)
    required = ((risk_ticker,) if defensive_ticker is None else
                (risk_ticker, defensive_ticker))

    def target(frames: dict[str, pd.DataFrame]):
        aligned = _aligned_frames(frames, required)
        today = aligned[risk_ticker].index[-1]
        known = state.loc[state.index <= today]
        if known.empty:
            return None
        return risk_ticker if bool(known.iloc[-1]) else defensive_ticker

    return _attach_source_metadata(target, risk_on)


def cap_pre_inception_history(
        series: pd.Series, first_tradeable_date, *,
        max_years_before: int) -> pd.Series:
    """Trim an external series to at most ``max_years_before`` years before
    ``first_tradeable_date``.

    An EXPANDING-window statistic (mean, std, z-score, ...) computed over
    a series that predates the traded instrument's own inception by
    decades can desensitize completely by the time trading is even
    possible -- see ``mining.constants.MAX_PRE_INCEPTION_BURN_IN_YEARS``
    for the rationale for the cap. This
    function only trims the INPUT; it adds no new statistic and changes
    no other behavior of whatever consumes its output.
    """
    if max_years_before <= 0:
        raise ValueError("max_years_before must be positive")
    source_attrs = dict(series.attrs)
    cutoff = pd.Timestamp(first_tradeable_date) - pd.DateOffset(years=max_years_before)
    trimmed = series.loc[pd.DatetimeIndex(series.index) >= cutoff]
    trimmed = trimmed.copy()
    trimmed.attrs.update(source_attrs)
    return trimmed


def lag_boolean_regime_to_availability(
        regime: pd.Series, traded_calendar: pd.DatetimeIndex, *,
        lag_trading_days: int) -> pd.Series:
    """Re-date an already-computed boolean regime to causal availability.

    Each entry becomes available at the ``lag_trading_days``-th trading
    session STRICTLY AFTER its own index date -- the same lag-application
    idiom as ``finra_margin_debt_growth_available_series``'s inner loop,
    generalized to any already-computed regime series regardless of its
    own date cadence (monthly, weekly, ...). The caller is responsible for
    anchoring each entry at the correct REFERENCE date before calling this
    (e.g. month-END for a monthly series dated at month-start).

    If two or more reference dates would become available on the exact
    same trading session (this happens whenever ``traded_calendar`` starts
    well after some of ``regime``'s own history -- e.g. a data series that
    predates the traded instrument's own inception, where every
    pre-inception reading is equally "not yet tradeable" until the same
    first real session), the LATEST reference date wins and earlier ones
    are dropped. This matches the "keep the last/most-recent" duplicate
    convention already used throughout this module and
    ``external_data.fetch_fred_series`` -- by the time a session is
    actually tradeable, only the freshest known reading is the true
    state; a stale superseded reading carries no separate information.
    """
    if lag_trading_days <= 0:
        raise ValueError("lag_trading_days must be positive")
    source_attrs = dict(regime.attrs)
    regime = pd.Series(regime, copy=True)
    regime.index = pd.DatetimeIndex(pd.to_datetime(regime.index)).tz_localize(None)
    if regime.index.duplicated().any():
        raise ValueError("regime series has duplicate reference dates")
    regime = regime.sort_index()
    calendar = pd.DatetimeIndex(pd.to_datetime(traded_calendar)).tz_localize(None)
    calendar = calendar.sort_values().unique()
    available = {}
    for reference_date, value in regime.items():
        later_sessions = calendar[calendar > reference_date]
        if len(later_sessions) < lag_trading_days:
            continue
        available_date = later_sessions[lag_trading_days - 1]
        available[available_date] = bool(value)  # later reference_date overwrites
    result = pd.Series(
        available, dtype=bool, name=regime.name or "risk_on").sort_index()
    result.attrs.update(source_attrs)
    return result


def finra_margin_debt_growth_available_series(
        monthly_balances: pd.Series, traded_calendar: pd.DatetimeIndex, *,
        lookback_months: int, lag_sessions: int) -> pd.Series:
    """Calendar-month FINRA growth dated at its causal trading availability.

    A reference month becomes available on the exact ``lag_sessions``-th
    session strictly later than month-end. Missing prior calendar months are
    never replaced with the previous observation.
    """
    if lookback_months <= 0 or lag_sessions <= 0:
        raise ValueError("lookback_months and lag_sessions must be positive")
    balances = _clean_series(monthly_balances)
    periods = balances.index.to_period("M")
    if periods.duplicated().any():
        raise ValueError("FINRA balances contain duplicate calendar months")
    by_month = pd.Series(balances.to_numpy(), index=periods, dtype=float)
    calendar = pd.DatetimeIndex(pd.to_datetime(traded_calendar)).tz_localize(None)
    calendar = calendar.sort_values().unique()
    available = {}
    for period, current in by_month.items():
        prior_period = period - lookback_months
        if prior_period not in by_month.index:
            continue
        prior = float(by_month.loc[prior_period])
        if prior <= 0:
            continue
        reference_end = period.to_timestamp("M")
        later_sessions = calendar[calendar > reference_end]
        if len(later_sessions) < lag_sessions:
            continue
        available_date = later_sessions[lag_sessions - 1]
        if available_date in available:
            raise ValueError("multiple FINRA months map to one availability session")
        available[available_date] = (float(current) - prior) / prior
    return pd.Series(
        available, dtype=float, name="finra_margin_debt_growth").sort_index()


def finra_margin_debt_risk_on_series(
        monthly_balances: pd.Series, traded_calendar: pd.DatetimeIndex, *,
        lookback_months: int, lag_sessions: int,
        growth_threshold: float) -> pd.Series:
    """FINRA risk-on state using the catalog's strict growth threshold.

    The mechanism says *positive/above-threshold* growth is risk-on, so an
    observation exactly equal to ``growth_threshold`` is risk-off.
    """
    if (isinstance(growth_threshold, bool) or
            not np.isfinite(float(growth_threshold))):
        raise ValueError("growth_threshold must be finite numeric")
    growth = finra_margin_debt_growth_available_series(
        monthly_balances, traded_calendar,
        lookback_months=lookback_months, lag_sessions=lag_sessions)
    result = (growth > float(growth_threshold)).rename("risk_on")
    result.attrs.update(monthly_balances.attrs)
    return result


# ---------------------------------------------------------------------------
# sector_dispersion_regime_cyclical_stance (catalog parameters:
# traded_ticker='XLY', dispersion_panel=the 9 original Select Sector SPDRs,
# dispersion_smoothing_days=20, regime_lookback_days=252,
# rebalance='on_regime_change').
#
# The "dispersion panel statistic" (`sector_dispersion`) is the
# cross-sectional return dispersion (RD) measure documented by Stivers &
# Sun (2010, JFQA 45(4): 987-1014): the cross-sectional standard deviation,
# across a panel of instruments, of that day's own simple return. Stivers &
# Sun find RD is a leading COUNTERCYCLICAL state variable -- low RD
# precedes stronger momentum/procyclical premia, high RD precedes stronger
# value/countercyclical premia. `dispersion_regime` classifies each date's
# SMOOTHED dispersion (a `smoothing_days` rolling mean, to damp single-day
# noise) against its OWN trailing `lookback_days` rolling mean into 'low'
# or 'high'. `make_sector_dispersion_cyclical_stance_signal_fn` gates a
# single cyclical sector's exposure (XLY in the catalog entry, but the
# factory is instrument-agnostic -- it trades whatever ticker's frame the
# engine hands the returned function) on that regime: long only while
# 'low', flat ('high') otherwise. This is a genuinely different mechanism
# shape from every other regime-gated candidate in this catalog (e.g.
# defensive_sector_recession_signal_rotation, credit_spread_hy_ig_risk_
# regime): those gate on a LEVEL/trend of a single ratio or spread; this
# one gates on a SECOND-MOMENT cross-sectional statistic computed across a
# multi-instrument panel, which is why it needs its own reusable panel
# statistic rather than reusing an existing single-series regime check.
# ---------------------------------------------------------------------------

DISPERSION_PANEL = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")
# The 9 ORIGINAL Select Sector SPDR ETFs (all inception 1998-12-16).
# XLRE (2015) and XLC (2018) are deliberately excluded from the panel --
# including them would either truncate the whole dispersion series to a
# post-2018 start or require a panel whose membership changes over time,
# which this fixed-column `panel_returns` statistic does not support (a row
# with any NaN member computes to NaN).

DISPERSION_SMOOTHING_DAYS = 20  # catalog parameters.dispersion_smoothing_days
REGIME_LOOKBACK_DAYS = 252  # catalog parameters.regime_lookback_days


def sector_dispersion(panel_returns: pd.DataFrame) -> pd.Series:
    """Cross-sectional return dispersion (Stivers & Sun 2010): for each
    date, the sample standard deviation (ddof=1) across columns of that
    date's own simple return. `panel_returns` must have at least 2
    columns (one per panel instrument) sharing a common index -- a
    cross-sectional statistic is undefined for a single series, so fewer
    than 2 columns raises ValueError rather than silently returning a
    degenerate (all-zero or all-NaN) series. A row with ANY NaN value
    computes to NaN (skipna=False) -- an incomplete cross-section on a
    given date is a data-alignment problem to surface, not to silently
    average over a partial panel."""
    if panel_returns.shape[1] < 2:
        raise ValueError(
            f"sector_dispersion needs >=2 columns (a panel), got {panel_returns.shape[1]}")
    return panel_returns.std(axis=1, ddof=1, skipna=False)


def dispersion_regime(dispersion: pd.Series,
                       smoothing_days: int = DISPERSION_SMOOTHING_DAYS,
                       lookback_days: int = REGIME_LOOKBACK_DAYS) -> pd.Series:
    """Classifies each date into the 'low' or 'high' dispersion regime.
    `dispersion` (e.g. `sector_dispersion`'s output) is first smoothed
    with a trailing `smoothing_days`-day rolling mean (damps single-day
    noise), then that smoothed value is compared to ITS OWN trailing
    `lookback_days`-day rolling mean. 'low' when the smoothed value is
    STRICTLY below its own trailing long-run average; 'high' when at or
    above (an exact tie is a documented tie-break toward 'high', i.e.
    toward exiting/staying flat, not toward holding a cyclical position
    on an ambiguous read). Both rolling windows require a FULL window of
    real observations (`min_periods` equals the window length) -- a date
    before either window is full classifies as NaN (object dtype, so NaN
    is distinguishable from the string 'high'), never a guessed regime."""
    smoothed = dispersion.rolling(smoothing_days, min_periods=smoothing_days).mean()
    long_run = smoothed.rolling(lookback_days, min_periods=lookback_days).mean()
    regime = pd.Series(np.nan, index=dispersion.index, dtype=object)
    valid = smoothed.notna() & long_run.notna()
    regime.loc[valid] = np.where(smoothed.loc[valid] < long_run.loc[valid], "low", "high")
    return regime


def make_sector_dispersion_cyclical_stance_signal_fn(
        panel_returns: pd.DataFrame,
        smoothing_days: int = DISPERSION_SMOOTHING_DAYS,
        lookback_days: int = REGIME_LOOKBACK_DAYS):
    """Returns a SignalFn closed over a pre-fetched, date-aligned
    `panel_returns` (daily simple returns across `DISPERSION_PANEL`, or
    an equivalent panel supplied by the caller/tests). The regime is
    computed ONCE up front over the full panel history -- each date's
    regime label depends only on that date and earlier ones (backward-
    looking rolling means), so precomputing it outside the per-bar
    signal_fn introduces no look-ahead. The returned function trades
    whatever instrument's OHLC frame the engine hands it (XLY per the
    catalog entry): enters long when today's regime is 'low', exits (or
    stays flat) otherwise -- including when today's date has no regime
    label at all (not yet in `panel_returns`' index, or before either
    rolling window fills), which is treated the same as 'high' (flat),
    never as "stay in whatever state we were already in"."""
    dispersion = sector_dispersion(panel_returns)
    regime = dispersion_regime(dispersion, smoothing_days, lookback_days)

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        today = frame.index[-1]
        today_regime = regime.loc[today] if today in regime.index else np.nan
        if pd.isna(today_regime):
            return SignalResult(entry=False, exit=True)
        want_long = today_regime == "low"
        return SignalResult(entry=want_long, exit=not want_long)

    return signal_fn
