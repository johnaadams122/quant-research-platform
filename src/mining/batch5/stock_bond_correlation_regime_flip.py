"""Batch5 candidate: stock_bond_correlation_regime_flip (single_long).

Rolling SPY/TLT return-correlation regime timing SPY. Source and directional
hypothesis (from the published sources cited below):

    In 2022 the SPX/AGG (stock/aggregate-bond) rolling correlation spiked to
    roughly +0.50, historically the highest in the modern sample, driven by
    the Fed's aggressive tightening cycle -- both equities and bonds fell
    together for the first time since 1977 (AQR, "A Changing Stock-Bond
    Correlation", https://www.aqr.com/Insights/Research/Journal-Article/
    A-Changing-Stock-Bond-Correlation; general market commentary
    corroborating the same 2022 episode).

    Academic grounding for WHY the sign flips: Campbell, Pflueger & Viceira
    (2020, Journal of Political Economy; also see Cieslak, NBER Working
    Paper 28184, "Common Shocks in Stocks and Bonds") show the stock-bond
    correlation's sign depends on the covariance between inflation shocks
    and growth shocks -- countercyclical inflation news (bad inflation in
    bad growth times) produces NEGATIVE stock-bond correlation (bonds hedge
    stocks, the traditional 60/40 logic); procyclical/monetary-shock-driven
    inflation news produces POSITIVE correlation (bonds and stocks fall
    together, no hedge benefit). This model explains roughly 70% of
    long-run variation in the US stock-bond correlation per that research.

    Practical takeaway used as this candidate's directional hypothesis: a
    rolling equity-Treasury return correlation that turns UNUSUALLY
    POSITIVE signals the market has entered (or is in) a monetary-policy-
    shock/inflation-shock regime -- historically a worse backdrop for
    equity returns than the more typical negative-correlation, growth-shock
    regime -- so reducing equity exposure while this regime persists is a
    directionally motivated, falsifiable bet, not an arbitrary threshold
    rule.

Rule
----
    Each trading day t on the common SPY/TLT calendar, compute a trailing
    W-trading-day Pearson correlation of SPY's and TLT's own daily log
    returns (contemporaneous, i.e. the reading for day t includes day t's
    own SPY_t/TLT_t returns -- exactly the academic literature's own
    realized-correlation convention, per Campbell/Pflueger/Viceira and
    Cieslak above). risk_on (hold SPY) while corr_t <= C; risk_off (flat
    cash) while corr_t > C. Re-evaluated every trading day.

W and C: chosen from the correlation distribution, not from P&L
----------------------------------------------------------------
W and C are set from the shape of the rolling SPY/TLT return-correlation
series itself (no fitting to backtest returns). A small grid of windows
W in {40, 60, 90} trading days gives the same qualitative picture: a
distribution centred well below zero (the "traditional 60/40 hedge" regime
the source literature describes as typical) with a right tail that only
occasionally reaches meaningfully positive readings, so the choice among
them is not fitting a shape that only one window produces. W=60 (roughly a
calendar quarter) is the common institutional convention and sits between
the noisiest (W=40) and most smoothed (W=90) choices.

C is placed near the upper tail of W=60's own empirical distribution
(roughly its 95th percentile, rounded to a clean C=0.30), so that only the
small share of trading days with an unusually positive stock-bond
correlation fall into the risk-off regime. The threshold marks where the
empirical tail begins; it is not reverse-engineered from any trade
sequence (the same discipline
bitcoin_extreme_decline_spillover_riskoff.py's own PARAMETERS comment
describes). The raw count of flag transitions and the engine's realized
trade count need not match exactly: the engine enforces its own one-bar
entry-then-hold mechanics (an entry filled at bar i is never exit-checked
until bar i+1, per ``run_backtest``'s own loop), which can fold two flag
transitions that fall close together into one realized trade.

The cited AQR figure uses SPX against an aggregate-bond index (AGG), not
TLT's long-duration Treasury exposure, and very likely a different window
and frequency, so a SPY/TLT reading is not expected to match it exactly;
only the direction of the regime story is shared. By the mechanism's own
logic, threshold crossings should cluster in inflation/monetary-shock
regimes (Fed tightening cycles) rather than in growth-shock stress
episodes, where bonds typically rally while stocks fall and the
correlation is deeply negative.

Design (the four non-negotiable properties):

* NO LOOK-AHEAD -- SPY's and TLT's own daily log returns are each computed
  on that instrument's OWN calendar first (``np.log(close).diff()``, so a
  return is never computed across a gap introduced by intersecting with a
  DIFFERENT instrument's calendar), THEN aligned to the common calendar
  (matching oil_price_change_equity_underreaction.py's own alignment
  convention). The rolling correlation for day t uses the trailing W
  common-calendar returns ENDING AT t (pandas ``.rolling(W).corr()``,
  which by construction never reads a row past its own current position);
  day t's own SPY_t/TLT_t returns are included (the source literature's own
  contemporaneous-correlation convention), but day t+1 and later are never
  read. The signal function only ever receives ``frame.iloc[:i+1]`` (the
  engine's own structural look-ahead guard) and independently only reads
  precomputed values through the bucket position it is asked about -- see
  this module's own test file for a direct look-ahead probe (bars after the
  decision bar are perturbed and the earlier signals must not change), not
  just an argument from the contract.
* PERFORMANCE -- the returned signal_fn does one numpy array lookup by
  position plus one scalar date check; the rolling correlation itself is
  computed ONCE in the factory, never recomputed per bar.
* HOLDOUT-BLIND -- both frames are truncated to <= the last bar of
  ``index`` before anything is computed, so a holdout bucket's signals
  cannot depend on data after the bucket; history BEFORE the bucket is
  kept as warm-up (the W-day trailing window the first in-bucket decision
  needs).
* DIRECTION -- always long SPY or flat cash; never short, never trades TLT.

Engine mapping (mining.backtest_engine.run_backtest), matching
vix_term_structure_carry.py's own daily level-crossing convention exactly:
entry[t] = corr_t <= C fires on bar t (fills at bar t+1's open, the
earliest the engine can buy); exit[t] = corr_t > C fires on bar t (fills
at bar t's own close). The engine only reads ``entry`` while flat and only
reads ``exit`` while long, so both flags can be computed independently,
per bar, straight from that bar's own correlation reading -- no lag beyond
the engine's own next-open/same-close fill semantics is needed or applied
(unlike the rotation-shape BTC-spillover family, which needs an explicit
lag because its target_fn returns which asset to hold as of the SAME
truncated frame; this SignalResult-based single_long shape already gets
the lag for free from the engine's own fill rule).

Interpretation choices (left to the implementer):

1. Boundary: corr_t == C exactly is risk-ON (hold), matching the rule's own
   "<=" on the entry side and ">" (strict) on the exit side -- the two
   conditions partition every finite, computable correlation reading; there
   is no third state.
2. Undefined signal (warm-up period before W common-calendar return
   observations exist, or a non-finite correlation reading) means NO
   SIGNAL that day (entry=False, exit=False) -- matches this catalog's own
   "undefined is not flat by rule, it is simply no signal today" convention
   (see vix1d_vix9d_inversion_deleverage_trigger.py's docstring), so the
   engine simply holds whatever position it already has (or stays flat
   pre-warm-up). This also covers SPY bars entirely before TLT's own
   history begins (TLT inception 2002-07-30): those bars are silent, the
   same "dates_before_signal_history_stay_flat" convention
   oil_price_change_equity_underreaction.py already established.
3. Duplicate dates in either close series keep the LAST occurrence before
   sorting, matching the de-duplication convention already used elsewhere
   in this batch.

NOT REACHABLE via the bespoke external-kwarg driver pattern, and
DELIBERATELY SO: this family uses the standard batch5 ``make_signal_fn
(ticker, index, frames)`` contract (three positional args, no extra
kwargs) -- confirmed against ``scripts/run_abi1_mining.py``'s own
``_build_curve`` function, which for every ``SINGLE_SHAPES`` family calls
``module.make_signal_fn(symbol, full_frame.index, module_frames)``
positionally with ``module_frames = {s: frames[s] for s in [symbol] +
signal_symbols}``. TLT is a normal ``data_pipeline.fetch_ohlc``-fetchable
ticker (unlike Kraken BTC spot or FRED series), so this family needs NO
bespoke driver script -- it is scoreable through the SAME generic dispatch
already used for oil_price_change_equity_underreaction.py and
vix_term_structure_carry.py.

DISTINCTNESS NOTE (verbatim mechanism text from the closest existing
catalog neighbors, ``data/mining/catalog.json``. No other catalog family
computes a rolling CORRELATION between SPY's and TLT's own RETURNS: the
other mechanism/distinctness_note fields that mention "correlat" together
with "TLT"/"bond"/"treasury" describe unrelated signals --
``real_rate_gold_equity_rotation`` [TIP/GLD],
``vol_managed_smallcap_leverage_switch`` [TNA/IWM realized variance],
``move_vix_divergence_of_fear_gauges_derisking`` [standardized MOVE/VIX
implied-volatility divergence] and ``cross_asset_financial_turbulence_
derisking`` [a multi-asset Mahalanobis turbulence index] -- none of them
is a pairwise stock-bond return correlation):

* ``yield_curve_slope_duration_rotation`` (catalog, status=eligible) --
  "Compute the daily price ratio TLT/SHY as a proxy for the Treasury
  yield-curve slope ... Take the ratio's 126-trading-day (6-month) rate of
  change. When that change exceeds +8% (curve flattening sharply) ..."
  A TLT/SHY PRICE RATIO trend on the yield curve's SLOPE, not a
  correlation of anyone's RETURNS, and it never touches SPY's own price
  series at all.
* ``credit_spread_hy_ig_risk_regime`` (catalog, status=eligible) --
  "Compute the daily price ratio HYG/LQD as a real-time proxy for the
  high-yield-vs-investment-grade credit spread ... Take a 20-day and a
  100-day simple moving average of this ratio." An HYG/LQD PRICE RATIO
  moving-average crossover on the CREDIT spread, not a correlation, and it
  never touches TLT or SPY at all.
* ``treasury_duration_volatility_timing`` (catalog, status=eligible) --
  "Compute TLT's trailing 20-day realized annualized volatility against
  its own trailing 252-day median. Below median (calm) -> hold long-
  duration TLT; above median ... -> rotate to short-duration SHY." A
  SINGLE-INSTRUMENT realized-VOLATILITY regime on TLT alone -- no SPY
  involvement, and volatility is not correlation.
* ``usd_strength_risk_regime`` (catalog, status=eligible) -- "Track UUP's
  closing price against its own 200-day simple moving average ... When UUP
  closes above its 200-day average ... move out of SPY into a defensive
  cash/short-Treasury (SHY) allocation." A currency TREND-following
  signal (UUP vs its own moving average); never computes a correlation and
  never involves TLT.
* ``cross_asset_time_series_momentum`` (catalog, status=eligible) --
  "Apply a trailing-return trend-following rule (long if trailing 12-month
  return positive, flat/short otherwise) to non-equity-index ETFs --
  Treasuries (TLT), gold (GLD), broad commodities (DBC)." A plain
  trailing-return MOMENTUM sign rule applied to TLT ON ITS OWN (one leg of
  a diversified sleeve) -- never a relationship between SPY and TLT, let
  alone their correlation.
* ``gold_long_treasury_relative_momentum_rotation`` (catalog,
  status=eligible) -- "each month hold whichever of GLD (physical gold) or
  TLT (20+ year Treasuries) had the higher trailing 12-month total
  return." A RELATIVE-MOMENTUM rotation BETWEEN GLD and TLT (both safe-
  haven legs, never SPY, never traded) -- SPY is not even one of the two
  assets, and the comparison is trailing return magnitude, not a
  correlation coefficient.

None of the above measures the actual statistical co-movement (a Pearson
correlation coefficient) between equity and Treasury RETURNS the way this
family does; every existing TLT-touching family uses TLT's own price
level, price ratio, trailing return, or realized volatility, never a
joint-return correlation with SPY.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult

FAMILY_ID = "stock_bond_correlation_regime_flip"
SHAPE = "single_long"
TRADED = ["SPY"]
SIGNAL = ["TLT"]
SNAPSHOTS = {}
BENCHMARK = "SPY"
# Copied byte-for-byte from the candidate JSON "parameters" field (driver
# drift check via json.dumps(sort_keys=True)).
PARAMETERS = {
    "signal": "spy_tlt_rolling_return_correlation",
    "traded_ticker": "SPY",
    "signal_ticker": "TLT",
    "correlation_window_days": 60,
    "risk_off_correlation_threshold": 0.3,
    "risk_off_condition": "rolling_correlation > threshold",
    "risk_on_condition": "rolling_correlation <= threshold",
    "flat_position": "cash",
    "rebalance": "daily",
}
# Sanity floor on the full-history trade count: the batch driver refuses a
# run that produces fewer trades than this (a guard against a silently
# degenerate signal), the same conservative-floor convention as every
# sibling family in this catalog.
EXPECTED_MIN_TRADES = 20

_WINDOW = int(PARAMETERS["correlation_window_days"])
_THRESHOLD = float(PARAMETERS["risk_off_correlation_threshold"])


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
    close = pd.Series(frame["Close"].to_numpy(), index=_naive_index(frame.index))
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close[close.index <= last_bar]


def spy_tlt_rolling_correlation(spy_close: pd.Series, tlt_close: pd.Series,
                                window: int) -> pd.Series:
    """Trailing ``window``-day Pearson correlation of SPY's and TLT's own
    daily log returns, indexed on their COMMON calendar.

    Each series' log return is computed on its OWN calendar first
    (``np.log(close).diff()``), so a return is never computed across a gap
    introduced by aligning to the OTHER instrument's calendar -- only after
    both return series exist are they intersected (matching
    oil_price_change_equity_underreaction.py's own alignment convention).
    The rolling correlation at position t is computed from the trailing
    ``window`` common-calendar returns ENDING AT t inclusive (pandas
    ``.rolling(window).corr()``), so it structurally cannot read a return
    past its own current row -- no look-ahead. NaN wherever fewer than
    ``window`` common-calendar return observations are available yet, or
    either return is non-finite within the window."""
    if window <= 1:
        raise ValueError(f"{FAMILY_ID}: window must be > 1, got {window}")
    spy_ret = np.log(spy_close.astype(float)).diff()
    tlt_ret = np.log(tlt_close.astype(float)).diff()
    common = spy_ret.index.intersection(tlt_ret.index).sort_values()
    spy_common = spy_ret.reindex(common)
    tlt_common = tlt_ret.reindex(common)
    return spy_common.rolling(window, min_periods=window).corr(tlt_common)


def correlation_regime_flags(bucket: pd.DatetimeIndex, corr: pd.Series,
                             threshold: float) -> tuple[pd.Series, pd.Series]:
    """Precompute (entry, exit) boolean Series aligned to ``bucket``.

    entry[t] = corr_t <= threshold (risk-on, hold SPY); exit[t] = corr_t >
    threshold (risk-off, flat cash). A bucket date with no correlation
    reading (before TLT's own history begins, within the W-day warm-up, or
    a non-finite value) is silent on both flags -- ``corr`` is looked up by
    date via ``reindex`` (no forward-fill), never by position, so a bucket
    date outside ``corr``'s own index is unambiguously "undefined", not
    accidentally inheriting a stale prior reading."""
    aligned = pd.to_numeric(corr.reindex(bucket), errors="coerce").to_numpy(dtype=float)
    computable = np.isfinite(aligned)
    entry = computable & (aligned <= threshold)
    exit_ = computable & (aligned > threshold)
    return (pd.Series(entry, index=bucket, name="entry"),
            pd.Series(exit_, index=bucket, name="exit"))


def make_signal_fn(ticker: str, index, frames):
    """Factory per the batch5 contract: ``index`` is the bucket the engine
    will run over; ``frames`` holds full-history OHLC for TRADED + SIGNAL."""
    if ticker != TRADED[0]:
        raise ValueError(f"{FAMILY_ID}: make_signal_fn expects ticker "
                         f"{TRADED[0]!r}, got {ticker!r}")
    spy_frame = _require_frame(frames, ticker)
    tlt_frame = _require_frame(frames, SIGNAL[0])

    bucket = _naive_index(index)
    if len(bucket) == 0:
        raise ValueError(f"{FAMILY_ID}: index must not be empty")
    if not bucket.is_monotonic_increasing or bucket.has_duplicates:
        raise ValueError(f"{FAMILY_ID}: index must be sorted and unique")
    last_bar = bucket[-1]

    spy_close = _clean_close(spy_frame, last_bar)
    tlt_close = _clean_close(tlt_frame, last_bar)

    corr = spy_tlt_rolling_correlation(spy_close, tlt_close, _WINDOW)
    entry_series, exit_series = correlation_regime_flags(bucket, corr, _THRESHOLD)
    entry_arr = entry_series.to_numpy(dtype=bool)
    exit_arr = exit_series.to_numpy(dtype=bool)
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
