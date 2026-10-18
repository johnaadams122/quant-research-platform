"""Batch5 candidate: betting_against_beta_low_vol (single_long).

Implements the pre-registered catalog rule below in the batch5
``make_signal_fn(ticker, index, frames)`` module contract.

Source (verbatim from the pre-registered catalog entry,
``data/mining/catalog.json``, ``family_id == "betting_against_beta_low_vol"``,
``added_date`` 2026-08-29): Frazzini, A. & Pedersen, L.H. (2014), "Betting
Against Beta", Journal of Financial Economics 111(1), 1-25.
https://econpapers.repec.org/RePEc:eee:jfinec:v:111:y:2014:i:1:p:1-25.

Mechanism (verbatim from the catalog): "Long low-beta equities/ETFs
(optionally leveraged toward unit beta), short or underweight high-beta ones
-- leverage-constrained investors bid up high-beta assets, compressing their
risk-adjusted return, so a low-beta-tilted portfolio earns a persistent
premium. Long-only variant: overweight low-volatility ETFs (e.g. SPLV,
USMV) relative to a cap-weighted benchmark." Catalog tickers are
``["SPLV", "USMV", "SPY"]``; SPLV (catalog ``tickers[0]``) is the traded leg and SPY is the
signal leg used only to compute SPLV's trailing beta -- USMV (catalog
``tickers[1]``) is a documented alternate proxy, not a second leg, and is
neither fetched nor traded here.

Rule
----
Each trading day, find the most recent MONTHLY rebalance date: the first
trading day whose (year, month) differs from the immediately preceding
trading day's. ``rebalance_dates`` (below) scans the FULL common SPLV/SPY
calendar once, vectorized, with no window limit at all --
``_MAX_TRADING_DAYS_PER_MONTH`` (a calendar month never spans more trading
days than that) is only a test-scoping constant (how far into a
restricted bucket the regression test checks), not a runtime limit in
this module. See ``rebalance_dates``'s own docstring for why the
unbounded full-calendar scan is required for correctness, not just an
optimization.
Compute SPLV's trailing beta against SPY (``PARAMETERS["beta_lookback_days"]
== 252`` trading days of daily returns) using SPLV's own returns only
THROUGH that rebalance position's date (never through "today") against
SPY's returns over the same calendar dates -- ordinary least-squares beta,
``cov(SPLV, SPY) / var(SPY)``. Hold SPLV (entry=True, exit=False) while that
beta reading is strictly below 1.0; go flat (entry=False, exit=True)
otherwise -- "otherwise" covers both "beta >= 1.0" and "beta undefined"
(insufficient history), i.e.
``hold = False if beta is undefined else bool(beta < _BETA_ENTRY_THRESHOLD)``.
This decision is RE-EVALUATED EVERY TRADING DAY using the LAST rebalance
date's beta reading, not recomputed daily from today's close -- the value
is frozen between rebalance dates and only updates when a new rebalance
position is reached (see ``test_signal_holds_constant_between_rebalance_
dates_even_as_raw_beta_drifts`` in this module's own test file for a direct
proof, not just an argument from the docstring).

Differences from this batch's ``stock_bond_correlation_regime_flip.py``
template (the two are the same SHAPE and the same factory-precomputes-
once-then-closure-indexes-by-position pattern, but NOT the same rule):

* CADENCE -- the template re-evaluates its correlation-vs-threshold test
  EVERY bar, straight off that bar's own rolling reading. This family
  re-evaluates only at MONTHLY rebalance dates and holds the reading
  constant on every other bar (see the rule above). The factory still
  precomputes an array ONCE (never per-bar), but that array is a per-bar
  LOOKUP of the OWNING rebalance date's beta reading, not a per-bar
  independent computation.
* THRESHOLD RULE -- the template compares a rolling CORRELATION to a
  threshold (entry on ``<=``, exit on strict ``>``, an inclusive/exclusive
  split with no third state). This family compares a rolling BETA to a
  threshold (entry on strict ``<``, exit on ``>=`` OR undefined), with no
  separate "no signal" state (see the WARM-UP note below).
* WARM-UP / UNDEFINED CONVENTION -- the template's own convention (shared
  with most of this batch) is "undefined means NO SIGNAL today" (entry=
  False, exit=False), leaving whatever position already exists untouched.
  This family's rule has no such third state: an undefined
  beta reading (warm-up, or insufficient overlapping history) resolves to
  ``hold=False`` exactly like a defined beta above threshold, i.e.
  entry=False, exit=True -- matching convertible_bond_moneyness_regime.py's
  own precedent that "False is also the engine-safe default" when a regime
  input is undefined.

What is THE SAME as the template (this batch5 module contract's own
invariants, ``make_signal_fn`` shape unchanged):

* Single traded ticker (SPLV) + exactly one signal ticker (SPY) -- the
  ordinary ``SINGLE_SHAPES`` two-frame case, reachable through
  ``scripts/run_abi1_mining.py``'s generic ``_build_curve`` dispatch (SPY is
  a normal ``data_pipeline.fetch_ohlc``-fetchable ticker, so -- like the
  template -- this family needs NO bespoke external-data driver script).
* NO LOOK-AHEAD -- everything the factory precomputes is derived from
  ``frame.iloc[:i+1]``-equivalent truncated history; the returned
  ``signal_fn`` only ever receives the engine's own structurally-truncated
  frame slice and independently only reads precomputed values through the
  bucket position it is asked about (see
  ``test_look_ahead_probe_future_bars_do_not_change_past_signals`` in this
  module's own test file).
* PERFORMANCE -- the returned ``signal_fn`` does one numpy array lookup by
  position plus one scalar date check; the rolling beta and the rebalance-
  position map are each computed ONCE in the factory, never recomputed
  per bar.
* HOLDOUT-BLIND -- both close series are truncated to <= the bucket's own
  last bar before anything is computed (``_clean_close``), so a holdout
  bucket's signals cannot depend on data after the bucket; history before
  the bucket is kept as warm-up (both the 252-day beta lookback and the
  full-calendar rebalance-date scan need it).
* DIRECTION -- always long SPLV or flat cash; never short, never trades
  SPY or USMV.

Interpretation choices (the catalog entry leaves these to the
implementer):

1. Duplicate dates in either close series keep the LAST occurrence before
   sorting, matching the de-duplication convention already used elsewhere
   in this batch (e.g. ``stock_bond_correlation_regime_flip.py``).
2. Beta's Cov/Var ratio is computed via pandas' own rolling ``.cov()``/
   ``.var()`` (both at the SAME window and hence the SAME implicit ddof,
   which cancels in the ratio -- the same ddof-invariance argument
   ``convertible_bond_moneyness_regime.py``'s own docstring makes for its
   sum-of-squares implementation), computed ONCE in the factory rather
   than as ``np.cov``/``np.var`` calls recomputed from scratch on every
   call; both formulas compute the same trailing OLS beta.
   A FULL ``beta_lookback_days``-length window of common-calendar returns
   is required (``min_periods=window``), matching every other beta-style
   module in this batch (e.g. ``convertible_bond_moneyness_regime.rolling_beta``),
   rather than admitting a partial (e.g. half-window) overlap. SPLV
   (inception 2011-05-05) and SPY share a normal, gap-free NYSE calendar,
   so the full-window requirement only lengthens the warm-up at the start
   of SPLV's own history.
3. A rebalance date that falls outside the SPLV/SPY common calendar (not
   expected for two NYSE-listed, currently-trading ETFs, but structurally
   possible for a corrupted or partial frame) looks up as an undefined
   beta reading via ``reindex`` returning NaN -- the same "no signal"-for-
   the-lookup consequence as an ordinary warm-up NaN, never a KeyError.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "betting_against_beta_low_vol"
SHAPE = "single_long"
TRADED = ["SPLV"]
SIGNAL = ["SPY"]
SNAPSHOTS: dict = {}
# ABI-1 always benchmarks against SPY, not a family's own catalog field --
# see any other batch5 module's own comment on this (e.g.
# treasury_duration_volatility_timing.py). Here it happens to equal the
# SIGNAL ticker too, which is incidental (SPY is used both as the beta
# regressor and as the hard ABI-1 benchmark), not a contract violation --
# TRADED and SIGNAL themselves do not overlap.
BENCHMARK = "SPY"
# Copied byte-for-byte from data/mining/catalog.json's
# "betting_against_beta_low_vol" entry (verified in
# test_batch5_betting_against_beta_low_vol.py via json.dumps(sort_keys=True)
# byte-identity against that file directly, same convention as
# treasury_duration_volatility_timing.py).
PARAMETERS = {
    "beta_lookback_days": 252,
    "rebalance": "monthly",
}
_BETA_LOOKBACK_DAYS = int(PARAMETERS["beta_lookback_days"])
_BETA_ENTRY_THRESHOLD = 1.0  # hold SPLV while trailing beta to SPY is below this
_MAX_TRADING_DAYS_PER_MONTH = 23  # a generous upper bound on a calendar
# month's trading-day span (incl. holidays). NOT used by rebalance_dates()
# below (that function scans the full calendar unbounded -- see its own
# docstring); kept only as the regression test's own scoping constant.

# Structural floor, not a measured result: SPLV is index-constructed to hold
# the S&P 500's lowest-volatility constituents, so its trailing beta to SPY
# is expected to sit below 1.0 most of the time and this monthly beta-regime
# rule can legitimately reduce to very few round trips. One trade is the
# smallest count that still proves the rule ever entered.
EXPECTED_MIN_TRADES = 1


def _naive_index(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_localize(None)
    return out.normalize()


def _require_frame(frames, symbol: str) -> pd.DataFrame:
    frame = frames.get(symbol) if isinstance(frames, dict) else None
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{FAMILY_ID}: frames must contain a non-empty OHLC "
                         f"DataFrame for {symbol}")
    if "Close" not in frame.columns:
        raise ValueError(f"{FAMILY_ID}: {symbol} frame is missing Close")
    return frame


def _clean_close(frame: pd.DataFrame, last_bar: pd.Timestamp) -> pd.Series:
    """Tz-naive, sorted, de-duplicated (keep last) Close series truncated to
    <= ``last_bar`` -- the holdout-blind step, done once per ticker."""
    close = pd.Series(pd.to_numeric(frame["Close"], errors="coerce").to_numpy(),
                       index=_naive_index(frame.index))
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close[close.index <= last_bar]


def rebalance_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """For every date in ``calendar``, the date of the most recent monthly
    rebalance at or before it -- the first trading day whose (year, month)
    differs from the immediately preceding trading day's.

    ``calendar`` is the FULL common SPLV/SPY trading calendar through the
    current call's last bar, never truncated to whatever sub-window is
    being scored -- so a bucket's true prior rebalance date is always
    resolvable even when it falls outside the bucket itself. That matters
    for correctness: a restricted, mid-history bucket -- e.g. a
    holdout-only window such as ``run_batch5_mining.py``'s own
    ``compute_candidate``/``_signal_factory`` passes for in-sample/holdout
    evaluation, as opposed to ABI-1's own full-history bucket -- that
    begins mid-month has NO preceding month-transition visible within it,
    so a scan limited to the bucket would treat the BUCKET'S OWN START as
    the rebalance date for up to ``_MAX_TRADING_DAYS_PER_MONTH`` bars
    instead of the true prior monthly rebalance before the bucket began.
    Only the very first calendar month of the ENTIRE traded history (SPLV's
    own inception window) has no true prior transition to find; that stretch
    falls back to position 0, an unavoidable universal warm-up identical to
    every other real-data family's own start-of-history warm-up, not an
    artifact of whatever bucket the engine happens to be scoring."""
    n = len(calendar)
    years = calendar.year.to_numpy()
    months = calendar.month.to_numpy()
    is_transition = np.zeros(n, dtype=bool)
    if n > 1:
        is_transition[1:] = (years[1:] != years[:-1]) | (months[1:] != months[:-1])
    transition_positions = np.nonzero(is_transition)[0]
    positions = np.arange(n)
    insert_at = np.searchsorted(transition_positions, positions, side="right") - 1
    have_transition = insert_at >= 0
    nearest_transition = np.where(
        have_transition, transition_positions[np.clip(insert_at, 0, None)], 0)
    return calendar[nearest_transition]


def trailing_beta_vs_spy(splv_close: pd.Series, spy_close: pd.Series,
                         window: int) -> pd.Series:
    """Trailing ``window``-day OLS beta of SPLV's daily simple returns
    against SPY's (``cov(SPLV, SPY) / var(SPY)``, the same ratio as an
    ``np.cov``/``np.var`` formula, ddof-invariant in the ratio -- see this
    module's docstring interpretation choice 2), indexed on their common
    calendar.

    Each series' return is computed on its OWN calendar first
    (``pct_change()``, simple returns), so a return is never computed across a gap introduced by
    aligning to the OTHER instrument's calendar -- only after both return
    series exist are they intersected. The reading at date t uses only the
    trailing ``window`` common-calendar returns ENDING AT t inclusive
    (pandas ``.rolling(window, min_periods=window)``), so it structurally
    cannot read a return past its own current row -- no look-ahead. NaN
    wherever fewer than a full ``window`` common-calendar return
    observations exist yet, or the trailing SPY variance is non-finite or
    not strictly positive (a degenerate denominator never manufactures a
    beta)."""
    if window <= 1:
        raise ValueError(f"{FAMILY_ID}: window must be > 1, got {window}")
    splv_ret = splv_close.pct_change()
    spy_ret = spy_close.pct_change()
    common = splv_ret.index.intersection(spy_ret.index).sort_values()
    splv_common = splv_ret.reindex(common)
    spy_common = spy_ret.reindex(common)
    cov = splv_common.rolling(window, min_periods=window).cov(spy_common).to_numpy(dtype=float)
    var = spy_common.rolling(window, min_periods=window).var().to_numpy(dtype=float)
    valid = np.isfinite(cov) & np.isfinite(var) & (var > 0.0)
    beta = np.divide(cov, var, out=np.full_like(cov, np.nan), where=valid)
    beta = np.where(valid, beta, np.nan)
    return pd.Series(beta, index=common)


def hold_flags(beta_at_rebalance: np.ndarray, threshold: float) -> tuple:
    """(entry, exit) boolean arrays from each bar's looked-up rebalance-date
    beta reading. entry[t] = beta is defined AND strictly below threshold
    (hold SPLV); exit[t] = NOT entry[t] (beta undefined OR beta >=
    threshold -- flat), i.e.
    ``hold = beta < threshold if beta is defined else False`` --
    unlike this batch's more common "undefined = no signal" convention
    (e.g. ``stock_bond_correlation_regime_flip.py``), this family's own
    mechanism has no third state: every bar is either long or flat,
    matching ``convertible_bond_moneyness_regime.py``'s own precedent that
    "False is also the engine-safe default" when a regime input is
    undefined."""
    beta = np.asarray(beta_at_rebalance, dtype=float)
    entry = np.isfinite(beta) & (beta < threshold)
    exit_ = ~entry
    return entry, exit_


def make_signal_fn(ticker: str, index, frames):
    """Factory per the batch5 contract: ``index`` is the bucket the engine
    will run over (SPLV's own calendar, per the generic single_long
    dispatch); ``frames`` holds full-history OHLC for TRADED + SIGNAL."""
    if ticker != TRADED[0]:
        raise ValueError(f"{FAMILY_ID}: make_signal_fn expects ticker "
                         f"{TRADED[0]!r}, got {ticker!r}")
    splv_frame = _require_frame(frames, ticker)
    spy_frame = _require_frame(frames, SIGNAL[0])

    bucket = _naive_index(index)
    if len(bucket) == 0:
        raise ValueError(f"{FAMILY_ID}: index must not be empty")
    if not bucket.is_monotonic_increasing or bucket.has_duplicates:
        raise ValueError(f"{FAMILY_ID}: index must be sorted and unique")
    last_bar = bucket[-1]

    splv_close = _clean_close(splv_frame, last_bar)
    spy_close = _clean_close(spy_frame, last_bar)

    beta_series = trailing_beta_vs_spy(splv_close, spy_close, _BETA_LOOKBACK_DAYS)
    # Scan the FULL common SPLV/SPY calendar (beta_series's own index, which
    # already spans from inception through last_bar, unrestricted by bucket)
    # for rebalance transitions -- NOT bucket alone -- so a bucket that
    # starts mid-month still resolves to its true prior rebalance date even
    # when that date falls before the bucket's own first bar. See
    # rebalance_dates()'s own docstring for why.
    full_calendar = beta_series.index
    reb_date_by_calendar_pos = rebalance_dates(full_calendar)
    bucket_calendar_pos = full_calendar.get_indexer(bucket)
    # A bucket date absent from the SPLV/SPY common calendar (not expected for
    # two NYSE-listed, currently-trading ETFs, but structurally possible for a
    # corrupted or partial frame) has no rebalance basis to look up -- treat
    # it the same as any other "undefined" reading via reindex-to-NaT below.
    reb_dates = pd.DatetimeIndex(
        np.where(bucket_calendar_pos >= 0,
                 reb_date_by_calendar_pos[np.clip(bucket_calendar_pos, 0, None)],
                 pd.NaT))
    beta_at_bucket = pd.to_numeric(
        beta_series.reindex(reb_dates), errors="coerce").to_numpy(dtype=float)
    entry_arr, exit_arr = hold_flags(beta_at_bucket, _BETA_ENTRY_THRESHOLD)
    bucket_values = bucket.asi8  # int64 ns for an O(1) scalar date check
    n_bucket = len(bucket)

    def signal_fn(frame: pd.DataFrame, i: int) -> SignalResult:
        if i < 0 or i >= n_bucket:
            raise IndexError(f"{FAMILY_ID}: bar position {i} outside the "
                             f"{n_bucket}-bar bucket this signal was built for")
        today = frame.index[-1]
        if pd.Timestamp(today).value != bucket_values[i]:
            raise ValueError(f"{FAMILY_ID}: bar {i} is {today}, but the bucket "
                             f"index has {bucket[i]} there -- the engine must "
                             "run over the same index handed to the factory")
        return SignalResult(entry=bool(entry_arr[i]), exit=bool(exit_arr[i]))

    return signal_fn
