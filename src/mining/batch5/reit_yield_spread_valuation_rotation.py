"""Batch5 candidate: reit_yield_spread_valuation_rotation
(rotation_two_asset).

Implements the catalogued mechanism from ``data/mining/catalog.json`` under
the ABI-1 batch5 module contract. Nothing about the mechanism, the
parameters, or the traded instruments is re-tuned here -- this implements
what the catalog entry specifies, resolving the prose's own genuine
ambiguities explicitly (see "Interpretation choices" below).

Catalog mechanism (verbatim, ``data/mining/catalog.json``, family_id
"reit_yield_spread_valuation_rotation", added_date 2026-08-29): "Long the
Vanguard Real Estate ETF (VNQ) when REITs are cheap relative to bonds --
the spread between the REIT sector's trailing dividend yield and the
10-year Treasury yield is wide relative to its own trailing history --
since historically a wide spread (roughly 1.5-2.0 points or more) has
preceded strong forward REIT total returns. Exit to cash/SPY when the
spread compresses well below its trailing average. Rebalance monthly
using a rolling z-score of the yield spread computed over a multi-year
lookback. Implementation note: signal_fn must fetch ^TNX via data_
pipeline.fetch_ohlc and compute VNQ's own trailing dividend yield from
its distribution history or a documented proxy, aligned by date." Source:
Nareit, "Using REIT Yield Spreads to Make Tactical Investment Decisions,"
Nareit Market Commentary, November 2020 (cited verbatim in the catalog
entry).

Rotation target: this module's ``make_target_fn`` is a THIN factory
wrapping ``mining.signals.make_external_regime_target_fn`` -- the SAME
factory ``defensive_sector_recession_signal_rotation.py`` already uses
for its own externally-derived (yield-spread) regime -- fed a precomputed
``pd.Series`` risk-on state. Unlike that sibling's series (sparse,
defined ONLY at month-end dates once warm), this module's series is
DENSE: defined at every bar from the z-score's own first-ever non-NaN
reading onward, carrying the standing hysteresis state forward
continuously between decision bars -- the pattern
``long_horizon_index_mean_reversion.standing_long_state`` uses (including
its correctness property: a bucket beginning mid-regime must resolve its
TRUE prior state, not merely wait for the next decision bar inside the
bucket to re-confirm a state that was already true -- see that module's
own docstring). ``signals.make_external_regime_target_fn``'s
own as-of lookup (``state.index <= today``) works correctly with EITHER a
sparse or a dense input series (defensive_sector_recession_signal_
rotation's own docstring "Design decision" section proves this for the
sparse case); this module simply supplies the dense form directly instead
of relying on that lookup to reconstruct it from a sparse one.

Interpretation choices (the catalog mechanism text is prose, not code;
every choice below is a genuine reading that is resolved and disclosed,
not silently picked)
--------------------------------------------------------------------------

1. VNQ's own trailing distribution yield is NOT obtainable from ``data_
   pipeline.fetch_ohlc`` (that function returns only dividend-ADJUSTED
   OHLC via ``auto_adjust=True``, no dividends column at all -- see
   ``src/data_pipeline.py:fetch_ohlc``).
   VNQ's own per-share cash dividends AND its own UNADJUSTED close are
   fetched together from ONE real network call,
   ``yfinance.Ticker("VNQ").history(period="max", auto_adjust=False)``
   (see ``fetch_vnq_dividend_inputs`` below) -- matching how this batch
   already fetches auxiliary data outside the standard OHLC path (e.g.
   ``stock_split_post_split_drift_pooled.fetch_split_events``'s own
   injectable-fetcher call to ``yfinance.Ticker(t).splits``). This is a
   deliberately LIGHTER architecture than ``preferred_stock_yield_spread_
   reversion.py``'s committed-data-file-plus-independent-source-
   reconciliation pipeline: that family's heavier machinery exists
   because Yahoo lacks five real PFF/PGX/PFFD distributions that
   independent publishers list. VNQ's own real distribution history was checked the
   same way (see "Real measurement 2" below) and shows no
   comparable defect -- a clean, gap-free quarterly cadence with zero
   Capital-Gains-type rows and zero splits -- so the lighter, direct-
   fetch architecture is proportionate here, not a shortcut around a
   known problem. Disclosed trade-off: unlike the committed-file siblings,
   this module has a REAL RUNTIME NETWORK DEPENDENCY in production (every
   real call to ``make_target_fn`` without an injected ``dividend_inputs``
   re-fetches VNQ's full history live) -- promoting this to a committed,
   provenance-tracked data file (matching ``preferred_stock_yield_spread_
   reversion.py``'s or ``ex_dividend_day_capture_pooled.py``'s own
   convention) is a reasonable hardening step if production
   reliability becomes a concern; it is out of scope for this module.

2. "Trailing 12-month dividend yield" = a rolling 252-TRADING-DAY SUM of
   VNQ's own cash dividends (the "Dividends" column only -- never
   "Capital Gains", matching ``preferred_stock_yield_spread_reversion``'s
   own "Data hygiene" item c guard, though measurement
   confirms VNQ carries zero Capital-Gains-type rows over its full real
   history, so nothing is actually filtered out in practice) divided by
   that session's own UNADJUSTED close, times 100 -- landing on the same
   percentage-point scale as ^TNX's Close (see item 4 below). A rolling-
   window SUM was chosen over a strict "trailing 4 quarterly dividends"
   count (the alternative reading) because it is more
   robust to a real skipped or irregularly-timed distribution, matching
   ``preferred_stock_yield_spread_reversion.trailing_distribution_yield``
   's own established convention. Measurement
   (see "Real measurement 2"): VNQ's real distribution history is in
   fact a clean, gap-free quarterly cadence (every real gap 80-107
   calendar days, exactly 4 events per complete calendar year), so a
   strict trailing-4-dividends count would have produced a nearly
   identical yield series here -- the rolling-window reading is chosen
   for consistency with this batch's established convention and future
   robustness, not because the two readings were measured to disagree.
   UNADJUSTED close (not the dividend-adjusted close ``data_pipeline.
   fetch_ohlc`` returns, which populates the engine's own ``frames["VNQ"]``
   used only for fills/"today" lookup here) is used for the SAME reason
   ``preferred_stock_yield_spread_reversion``'s own docstring choice 1
   gives: an adjusted close is back-adjusted for every FUTURE payout,
   which inflates past yields and biases a rolling z-score. A full
   252-session window is required (NaN before) -- VNQ's first year never
   shows an understated yield.

3. ``spread = VNQ_trailing_yield_pct - ^TNX.Close`` (both percentage
   points). ``^TNX``'s Close is used directly with NO rescale -- Real
   measurement 1 below reconfirms (does not merely assume) the same
   finding ``defensive_sector_recession_signal_rotation.py``'s own
   docstring already established for this exact ticker: yfinance's
   ``^TNX`` Close values are already directly comparable percentage-point
   yields.

4. Rolling z-score of the spread over ``PARAMETERS["yield_spread_lookback
   _years"] * 252`` = 756 trading days (full window required, ddof=1),
   with the SAME zero-dispersion float-noise guard every sibling
   z-score helper in this batch uses (e.g. ``preferred_stock_yield_
   spread_reversion.trailing_yield_zscore``, ``long_horizon_index_mean_
   reversion.rolling_zscore``) -- a ROLLING (not expanding) window of the
   spread's own recent history, matching the catalog's own word
   ("rolling z-score... computed over a multi-year lookback").

5. Hysteresis band, mirroring ``long_horizon_index_mean_reversion``'s own
   hysteresis design: enter/stay long VNQ ("REITs cheap") while
   ``z >= entry_zscore`` (0.5) has fired at a decision bar and has not
   since fallen to ``z <= exit_zscore`` (-0.5) or below; exit/stay in SPY
   ("REITs expensive" or compressed) once ``z <= exit_zscore`` (-0.5) has
   fired, until z rises back to 0.5 or above. Strictly inside the band,
   the prior state carries forward unchanged. Checked ONLY at decision
   bars (month-end, item 6 below); on every OTHER bar the state is
   simply carried forward -- see ``reit_risk_on_series`` for the exact
   dense, continuously-carried-forward step function this produces
   (matching ``long_horizon_index_mean_reversion.standing_long_state``'s
   own shape, generalized from a two-state LONG/FLAT single-asset rule to
   this family's two-asset VNQ/SPY rotation). An undefined (NaN) z
   encountered AT a decision bar resolves to the defensive leg (SPY) --
   the same "undefined = flat/defensive" convention
   -- though this should not occur in practice once the rolling window is
   full, since neither VNQ's own dividend/close series nor ^TNX's own
   Close series has a real internal gap over the history checked (see
   "Real measurement 2"). Before the
   z-score's OWN first-ever non-NaN reading (the initial 756-session
   warm-up), the state is left UNDEFINED (NaN), never resolved to a
   guessed default -- this is a DIFFERENT case from an undefined reading
   AT a decision bar, and is handled separately in ``reit_risk_on_series``:
   during warm-up there is no real information at all, and defaulting to
   SPY for that whole multi-year stretch would misrepresent "no signal yet" as a real decision.
   ``signals.make_external_regime_target_fn`` already drops every NaN
   entry and returns ``None`` (no position) when nothing is yet defined at
   or before "today" -- so the warm-up window is a real no-position period
   in the backtest, not a silently-chosen SPY default.

6. "Rebalance monthly" = decision bars are MONTH-END only -- the LAST
   trading day of the month (a bar is a decision bar iff the NEXT bar of
   the calendar falls in a different calendar month), the SAME convention
   every other monthly-rebalance sibling in this batch uses (e.g.
   ``vol_managed_intl_country_etf_pooled.py``, ``defensive_sector_
   recession_signal_rotation.py``, ``long_horizon_index_mean_reversion.
   py``). Decision bars are computed on the VNQ-yield/``^TNX`` common
   calendar (the spread series' own aligned calendar), not on the engine's
   separately-fetched ``frames["VNQ"]``/``frames["SPY"]`` calendar -- the
   two occasionally differ by a session around a bond-market-only holiday,
   which merely shifts by at most one session WHEN a new month's reading
   first becomes visible to the traded legs (the same disclosed choice
   ``defensive_sector_recession_signal_rotation.py``'s own docstring
   makes for the identical reason).

No look-ahead: every series here (``vnq_trailing_yield_pct``,
``yield_spread_series``, ``spread_zscore``, the month-end flags, the
hysteresis state) is a ROLLING or WALK-FORWARD computation whose value at
date t reads only data dated <= t; nothing is shifted backward and no
future close is ever read. The dense hysteresis state is precomputed ONCE,
over the FULL real fetched history, at ``make_target_fn`` build time (the
same "precompute once from full causal history, then as-of filter per
call" pattern ``defensive_sector_recession_signal_rotation.py`` also
uses) -- safe specifically because every value in
that precomputed series is itself causal, so the later per-call as-of
filter inside ``signals.make_external_regime_target_fn`` (``state.index <=
today``, where "today" is the TRUNCATED bucket's own last date at each
simulated decision point) can never expose a future date's already-
computed reading before that date has actually occurred in the truncated
frames the engine hands to ``target()``.

Real measurement 1 -- units (verified against fetched data, re-confirming
rather than assuming what ``defensive_sector_recession_signal_rotation.py``'s
docstring states for this exact ticker): ``^TNX`` closes are already in
the ~4-5 range a real 10-year Treasury yield should be in, directly
subtractable from VNQ's own computed percentage-point yield with no
rescale.

Real measurement 2 -- VNQ's dividend history (verified against a fetch of
``yfinance.Ticker("VNQ")``): VNQ's price-data history begins 2004-09-29;
its dividend-data history begins 2004-12-22 -- a measured ~3-month gap
between the two (disclosed, not silently reconciled -- the trailing-yield
rolling sum handles this correctly by construction: it is simply
zero-contribution from dividends before the first real ex-date, same as
any other pre-window period). Cadence check (mirroring
``preferred_stock_yield_spread_reversion``'s own "Data hygiene" diligence,
though not its full independent-source-reconciliation machinery -- see
interpretation choice 1 above for why that heavier machinery is not
warranted here): every gap between consecutive VNQ ex-dates is 80-107
calendar days (mean 91.3), every complete calendar year 2005-2025 carries
exactly 4 events, and VNQ's ``history()`` carries ZERO nonzero "Capital
Gains" rows and ZERO nonzero "Stock Splits" rows over its full history -- a
clean, gap-free quarterly distributor, not a data quality concern.

Real measurement 3 -- the spread/z-score history shows an economically
plausible wide-vs-narrow pattern, not an invented or assumed one: the
spread (VNQ trailing yield pct minus ^TNX Close) is first defined 252
trading sessions after VNQ's own price-data inception, and its widest
readings fall in the 2008-2009 financial-crisis era, when REIT prices
collapsed far faster than the 10-year yield fell -- a well-documented
"REITs cheap" episode, not an artifact. The 756-session rolling z-score is
first defined during the 2008 financial crisis itself, a sensible date for
this rule's own warm-up to first complete.

Trade-count floor: a monthly z-score rotation through
``mining.rotation_engine.run_two_asset_rotation`` produces a modest number
of completed segments. The boundary masking in ``reit_risk_on_series``
(see its docstring) keeps any bar between the
z-score's first-ever defined bar and the first real decision bar
undefined instead of a fabricated early decision; it only affects that
pre-decision boundary, never the decision-bar-gated entry/exit dates.
``EXPECTED_MIN_TRADES`` is a conservative sanity floor (the driver refuses
a run below it), matching this batch's own convention for a family whose
trade count could plausibly shift by a session or two as new data
arrives (e.g. ``defensive_sector_recession_signal_rotation.py``'s own
floor).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining import signals

FAMILY_ID = "reit_yield_spread_valuation_rotation"
SHAPE = "rotation_two_asset"
TRADED = ["VNQ", "SPY"]         # [risk, defensive]
SIGNAL = ["^TNX"]               # external price-frame ticker the generic
                                 # dispatch auto-loads -- the 10-year
                                 # Treasury yield proxy, never traded
BENCHMARK = "SPY"
# ABI-1 ALWAYS benchmarks the account against SPY, never a family's own
# catalog "benchmark" field -- see scripts/run_abi1_mining.py's
# MARKET_BENCHMARK and every sibling rotation module's own identical
# disclosure (e.g. defensive_sector_recession_signal_rotation.py). Here it
# happens to also be TRADED[1] (the defensive leg) and this family's own
# natural market reference, which is incidental, not a contract violation.
SNAPSHOTS: dict = {}
# Copied byte-for-byte from data/mining/catalog.json's real entry for this
# family_id (verified via json.dumps(sort_keys=True) byte-identity in this
# module's own test, which reads that file directly rather than
# duplicating a value here that could drift).
PARAMETERS = {
    "yield_spread_lookback_years": 3,
    "entry_zscore": 0.5,
    "exit_zscore": -0.5,
    "rebalance": "monthly",
}
# Conservative sanity floor on the VNQ-bound aligned calendar (see module
# docstring's "Trade-count floor" section), same convention as e.g.
# defensive_sector_recession_signal_rotation.py's own floor.
EXPECTED_MIN_TRADES = 10

_TRAILING_YIELD_DAYS = 252
_ZSCORE_WINDOW_DAYS = _TRAILING_YIELD_DAYS * int(PARAMETERS["yield_spread_lookback_years"])
_ENTRY_Z = float(PARAMETERS["entry_zscore"])
_EXIT_Z = float(PARAMETERS["exit_zscore"])


# ------------------------------------------------------------- helpers

def _naive_index(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_localize(None)
    return out.normalize()


def _require_frame(frames, symbol: str) -> pd.DataFrame:
    frame = frames.get(symbol) if isinstance(frames, dict) else None
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{FAMILY_ID}: frames must contain a non-empty "
                         f"DataFrame for {symbol}")
    if "Close" not in frame.columns:
        raise ValueError(f"{FAMILY_ID}: {symbol} frame is missing Close")
    return frame


def _clean_close(frame: pd.DataFrame) -> pd.Series:
    """Tz-naive, sorted, de-duplicated (keep last) Close series -- no
    positivity requirement, matching ``defensive_sector_recession_signal_
    rotation._clean_close``'s own convention for a SIGNAL-only yield
    ticker (``^TNX`` has no real historical negative prints the way
    ``^IRX`` does, but this module's own code never assumes that)."""
    index = _naive_index(frame.index)
    close = pd.Series(pd.to_numeric(frame["Close"], errors="coerce").to_numpy(),
                       index=index)
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close[close.notna()]


def fetch_vnq_dividend_inputs(fetcher=None) -> tuple:
    """Real network fetch of VNQ's OWN unadjusted Close + per-share cash
    dividends (``yfinance.Ticker("VNQ").history(auto_adjust=False)``) --
    external auxiliary data ``data_pipeline.fetch_ohlc`` cannot provide (it
    returns only dividend-adjusted OHLC via ``auto_adjust=True``, no
    dividends column). ``fetcher(ticker) -> DataFrame`` defaults to the
    real yfinance call (imported lazily so this module never needs
    yfinance as an import-time dependency) -- the same injectable-fetcher
    idiom ``stock_split_post_split_drift_pooled.fetch_split_events`` uses
    for ``Ticker.splits``; here ONE call returns both the unadjusted Close
    and a "Dividends" column together (see docstring interpretation
    choice 1). Only "Dividends" is read, never "Capital Gains" (matching
    ``preferred_stock_yield_spread_reversion``'s own "Data hygiene" item c
    guard) -- measurement: VNQ carries zero Capital-
    Gains-type rows and zero splits over its full real history, so this
    guard is a standing precaution, not a data edit.

    Returns ``(unadjusted_close, dividends)``, both tz-naive ``pd.Series``
    aligned to VNQ's own real trading calendar; ``dividends`` is 0.0 on
    every non-ex-date session, never NaN. The still-in-progress current
    session (an all-NaN Close row, same shape every other batch5 real-data
    fetch drops) is dropped before anything else."""
    if fetcher is None:
        import yfinance as yf

        def fetcher(ticker):  # noqa: E306
            return yf.Ticker(ticker).history(period="max", auto_adjust=False)

    raw = fetcher(TRADED[0])
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError(f"{FAMILY_ID}: {TRADED[0]} dividend/close fetch "
                         "returned no data")
    raw = raw.dropna(subset=["Close"]).copy()
    if raw.empty:
        raise ValueError(f"{FAMILY_ID}: {TRADED[0]} dividend/close fetch "
                         "returned no usable (non-NaN Close) rows")
    raw.index = _naive_index(raw.index)
    close = pd.Series(pd.to_numeric(raw["Close"], errors="coerce").to_numpy(),
                       index=raw.index)
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close = close[close.notna()]
    if "Dividends" in raw.columns:
        dividends = pd.Series(
            pd.to_numeric(raw["Dividends"], errors="coerce").to_numpy(),
            index=raw.index)
        dividends = dividends[~dividends.index.duplicated(keep="last")].sort_index()
    else:
        dividends = pd.Series(0.0, index=close.index)
    dividends = dividends.reindex(close.index).fillna(0.0)
    return close, dividends


# --------------------------------------------------------- computation

def vnq_trailing_yield_pct(unadjusted_close: pd.Series, dividends: pd.Series, *,
                           trailing_days: int = _TRAILING_YIELD_DAYS) -> pd.Series:
    """VNQ's own trailing-``trailing_days``-session SUM of cash dividends
    divided by that session's own UNADJUSTED close, times 100 -- see
    docstring interpretation choice 2 for the full reasoning (rolling-sum
    vs trailing-4-count, unadjusted vs adjusted close). NaN until a full
    ``trailing_days`` window of history exists; aligned to
    ``unadjusted_close.index``, a value at t uses only dividends dated
    <= t (causal)."""
    if trailing_days <= 0:
        raise ValueError(f"{FAMILY_ID}: trailing_days must be positive")
    close = pd.Series(unadjusted_close, copy=True)
    payouts = pd.Series(dividends, copy=True).reindex(close.index).fillna(0.0)
    trailing = payouts.rolling(trailing_days, min_periods=trailing_days).sum()
    return ((trailing / close) * 100.0).rename("vnq_trailing_yield_pct")


def yield_spread_series(vnq_yield_pct: pd.Series, tnx_close: pd.Series) -> pd.Series:
    """``VNQ_trailing_yield_pct - ^TNX.Close`` (both percentage points,
    see docstring interpretation choice 3) on the two series' common
    calendar (``vnq_yield_pct``'s OWN defined dates intersected with
    ``tnx_close``'s dates) -- causal by construction, no shift, no future
    read."""
    common = vnq_yield_pct.dropna().index.intersection(tnx_close.index).sort_values()
    if len(common) == 0:
        return pd.Series(dtype=float)
    return (vnq_yield_pct.reindex(common) - tnx_close.reindex(common)).rename(
        "yield_spread")


def spread_zscore(spread: pd.Series, *,
                  window: int = _ZSCORE_WINDOW_DAYS) -> pd.Series:
    """Rolling ``window``-session z-score of ``spread`` (full window
    required, ddof=1) -- the same rolling (not expanding) convention and
    zero-dispersion float-noise guard every sibling z-score helper in this
    batch uses (e.g. ``preferred_stock_yield_spread_reversion.trailing_
    yield_zscore``, ``long_horizon_index_mean_reversion.rolling_zscore``).
    A rolling std below the relative tolerance floor is treated as
    undefined (NaN), never a spurious near-infinite z."""
    if window < 2:
        raise ValueError(f"{FAMILY_ID}: window must be >= 2, got {window}")
    mean = spread.rolling(window, min_periods=window).mean()
    std = spread.rolling(window, min_periods=window).std(ddof=1)
    tolerance = 1e-9 * mean.abs() + 1e-15
    z = (spread - mean) / std.where(std > tolerance)
    return z.rename("yield_spread_zscore")


def _month_end_flags(index: pd.DatetimeIndex) -> np.ndarray:
    """True where the NEXT date in ``index`` falls in a different calendar
    month than the current one -- the last date is always False. Same
    convention as every other monthly-rebalance sibling in this batch
    (defined locally here per this batch's self-contained-module
    convention)."""
    months = index.year.to_numpy() * 12 + index.month.to_numpy()
    flags = np.zeros(len(index), dtype=bool)
    if len(index) > 1:
        flags[:-1] = months[1:] != months[:-1]
    return flags


def reit_risk_on_series(zscore: pd.Series, decision_flags: np.ndarray, *,
                        entry_z: float = _ENTRY_Z, exit_z: float = _EXIT_Z
                        ) -> pd.Series:
    """Dense per-bar risk-on state (1.0 = VNQ / 0.0 = SPY), aligned to
    ``zscore.index`` -- see docstring interpretation choice 5 for the full
    design reasoning.

    Checked ONLY where ``decision_flags`` is True (month-end bars) and
    CARRIED FORWARD unchanged on every other bar -- matching
    ``long_horizon_index_mean_reversion.standing_long_state``'s own dense,
    continuously-carried-forward step-function shape (a bucket beginning
    mid-regime must resolve its TRUE prior state, not merely wait for the
    next decision bar reached inside the bucket). At a decision bar: enter/stay VNQ at ``z >= entry_z``;
    exit/stay SPY at ``z <= exit_z``; strictly inside the band, hold
    whatever the state already is. A NaN z encountered AT a decision bar
    resolves to SPY (this family's own "undefined = flat/defensive"
    convention).

    Separately -- and this is the piece that is NOT in
    ``standing_long_state`` -- every bar STRICTLY BEFORE the first REAL
    decision bar (the first bar that is BOTH a month-end decision bar AND
    has a defined z-score) is masked to NaN in the RETURNED series,
    regardless of what the internal walk-forward state happens to be at
    that point (it starts at SPY/False by construction and would
    otherwise silently read as a real SPY decision for the entire
    multi-year warm-up, AND for any gap between the
    z-score's first-ever defined bar and the next real decision bar, since
    those two dates are not the same: the z-score can first become defined
    on a non-decision bar, weeks before the next month-end decision bar). This is
    intentionally different from an in-series undefined reading (handled
    above, resolves to SPY): before the FIRST REAL DECISION there is no
    real signal at all, and ``signals.make_external_regime_target_fn``
    (this module's own consumer) already treats a NaN entry as "not yet
    defined," dropping it and returning ``None`` (no position) rather
    than a guessed default -- see the module docstring's "Real
    measurement 3"/warm-up discussion for why that distinction matters
    here specifically."""
    values = pd.to_numeric(zscore, errors="coerce").to_numpy(dtype=float)
    n = len(values)
    if len(decision_flags) != n:
        raise ValueError(f"{FAMILY_ID}: zscore and decision_flags length "
                         f"mismatch ({n} vs {len(decision_flags)})")
    raw_state = np.zeros(n, dtype=bool)
    is_risk_on = False
    for i in range(n):
        if decision_flags[i]:
            z = values[i]
            if np.isnan(z):
                is_risk_on = False
            elif z >= entry_z:
                is_risk_on = True
            elif z <= exit_z:
                is_risk_on = False
            # else: strictly inside the band -- hold the prior state.
        raw_state[i] = is_risk_on

    # The z-score's own first-ever non-NaN reading can land on a bar that
    # is NOT itself a decision bar (rolling-window completion has nothing
    # to do with month-end timing). Masking only up to the first DEFINED
    # bar would expose `raw_state`'s internal SPY/False default as a
    # "real" decision on every bar in that gap, even though no decision
    # had been evaluated yet. The mask therefore runs through the first
    # bar that is BOTH defined AND a real decision bar.
    ever_defined = ~np.isnan(values)
    first_real_decision = int(np.argmax(ever_defined & decision_flags)) \
        if (ever_defined & decision_flags).any() else n
    out = np.where(np.arange(n) >= first_real_decision,
                   raw_state.astype(float), np.nan)
    return pd.Series(out, index=pd.DatetimeIndex(zscore.index))


# ------------------------------------------------------------- factory

def make_target_fn(frames: dict, *, dividend_inputs=None):
    """Rotation target factory (driver contract: called ONCE, positionally,
    as ``module.make_target_fn(module_frames)`` -- see ``scripts/run_abi1_
    mining.py``'s own single-argument call site. ``dividend_inputs`` is
    therefore an OPTIONAL keyword-only argument, never reachable from the
    real driver, used only for test injection of a synthetic ``(close,
    dividends)`` pair -- matching ``preferred_stock_yield_spread_
    reversion.make_signal_fn``'s own ``inputs=`` idiom).

    ``frames`` must carry TRADED[0] (VNQ), TRADED[1] (SPY) and SIGNAL[0]
    (``^TNX``) -- fails closed on any missing key. VNQ's and SPY's own
    frames are used only for presence / the "today" as-of lookup inside
    ``signals.make_external_regime_target_fn``'s returned closure; the
    yield itself never reads ``frames["VNQ"]`` -- it is computed from a
    SEPARATE real fetch of VNQ's own unadjusted close + cash dividends
    (see ``fetch_vnq_dividend_inputs`` and docstring interpretation choice
    1: ``data_pipeline.fetch_ohlc``, which populates ``frames["VNQ"]``,
    returns only dividend-adjusted OHLC and carries no dividends column at
    all)."""
    for key in (TRADED[0], TRADED[1], SIGNAL[0]):
        if key not in frames:
            raise ValueError(
                f"{FAMILY_ID}: make_target_fn requires a frame for {key}")
    _require_frame(frames, TRADED[0])
    _require_frame(frames, TRADED[1])
    tnx_frame = _require_frame(frames, SIGNAL[0])
    tnx_close = _clean_close(tnx_frame)

    if dividend_inputs is None:
        unadjusted_close, dividends = fetch_vnq_dividend_inputs()
    else:
        unadjusted_close, dividends = dividend_inputs

    yield_pct = vnq_trailing_yield_pct(unadjusted_close, dividends)
    spread = yield_spread_series(yield_pct, tnx_close)
    zscore = spread_zscore(spread)
    decision = _month_end_flags(pd.DatetimeIndex(spread.index))
    risk_on = reit_risk_on_series(zscore, decision)

    return signals.make_external_regime_target_fn(
        risk_on, risk_ticker=TRADED[0], defensive_ticker=TRADED[1])
