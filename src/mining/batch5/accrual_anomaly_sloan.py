"""Candidate: accrual_anomaly_sloan (shape pooled_single_asset).

Mechanism: Sloan (1996), "Do Stock Prices Fully Reflect Information in
Accruals and Cash Flows about Future Earnings?" -- rank equities by total
accruals scaled by (average) total assets; the LOW-accrual/high-cash-
flow-quality subset earns persistently higher subsequent returns than
the high-accrual subset, because investors overweight the less-
persistent accrual component of earnings and are systematically
surprised when it reverses. This module holds the pool's own LONG side
only (bottom accrual = highest quality), reformed annually.

Catalog note: the catalog entry notes that no real ETF/ETN isolates the
accrual signal alone, so this module ranks companies directly from real
EDGAR point-in-time XBRL fundamentals (`mining.edgar_data`,
`mining.edgar_point_in_time`). Total accruals is computed directly from
three well-tagged, common XBRL concepts (see `mining.edgar_data`'s own
`NET_INCOME_TAGS`/`CFO_TAGS`/`TOTAL_ASSETS_TAGS`):

    accrual_ratio = (NetIncomeLoss_annual - CFO_annual) / average_total_assets

This is the standard CASH-FLOW-STATEMENT-based accrual proxy (Hribar &
Collins 2002), not Sloan's original BALANCE-SHEET-CHANGE formula
(``deltaCA - deltaCash - deltaCL + deltaSTD - Dep``, all over average
total assets) -- that formula needs several additional XBRL concepts
(current assets, cash, current liabilities, short-term debt,
depreciation) whose tag-mapping this module does not implement
(a real, disclosed methodological simplification, not a silent
substitution -- Hribar & Collins found the two approaches diverge
mainly around M&A/discontinued-operations activity, otherwise tracking
closely). "Average total assets" = mean of the latest and PRIOR known
total assets (Sloan's own convention: Sloan 1996 scales by the mean of
beginning and ending total assets) when both are known; falls back to the latest known assets alone
for a company whose disclosure history does not yet reach back a second
fiscal year -- a disclosed approximation, not a coverage failure (see
`mining.accrual_coverage`'s own docstring).

Universe: the same CIK-resolved 25-ticker S&P 500 watchlist as
`capex_cycle_blue_chip_dip_pooled` (that module's `TRADED`, built by
`scripts/build_capex_watchlist.py`, diversified
industrials/tech/materials/utilities/energy blue chips), imported rather
than rebuilt. The rank is taken within these 25 names, not across the
whole S&P 500.

Selection is the bottom QUINTILE (`bottom_fraction=0.20`, a disclosed
default) of the members with a defined accrual ratio, where Sloan (1996)
uses deciles: with 10 to 25 defined names (see RULE), a quintile selects
two to five of them.

RULE
----
    bottom_fraction = 0.20       # of members with a DEFINED accrual ratio
    min_rank_members = 10        # floor before ranking is even defined
    at each REFORMATION bar (the last trading day of each calendar year
    present in the pool's own trading calendar):
      for each ticker T in the pool:
        accrual_T = latest known (NetIncomeLoss - CFO) / average_assets,
                    point-in-time, EDGAR filed-date gated (never
                    fiscal-period-end -- see mining.edgar_point_in_time)
        defined_T = accrual_T is known and not stale
                    (edgar_point_in_time.STALENESS_CAP_DAYS)
      n_defined = count of tickers with defined_T this reformation bar
      if n_defined < min_rank_members: ranking UNDEFINED this year --
        no entries or exits fire for ANYONE (never a spurious signal
        from a too-thin cross-section)
      else:
        n_select = max(1, round(bottom_fraction * n_defined))
        selected = the n_select tickers with the LOWEST accrual_T among
                   the defined ones (ties broken alphabetically by
                   ticker, deterministic)
      if T in selected and position==FLAT: BUY T at the OPEN of the
        next trading session
      if position==LONG and T NOT in selected (including "ranking
        undefined this year" or "T itself undefined"): SELL T at the
        CLOSE of that SAME reformation bar
      if T in selected and position==LONG: hold, no interim action --
        annual reformation is a re-evaluated commitment, not a fixed
        one-year hold
      one trade = one contiguous LONG holding period per ticker; every
        ticker's trades concatenated into one pooled sample, scored as
        ONE trial (matches every sibling pooled_single_asset family)

No time-stop: a name that remains in the bottom accrual quintile at every
reformation stays held indefinitely, matching the academic replication
convention of holding as long as membership persists.

Engine mapping (mining.backtest_engine.run_backtest): an entry=True on
the reformation bar fills at the NEXT bar's open; an exit=True fires ON
that SAME reformation bar and fills at ITS close -- IDENTICAL timing
convention to `capex_cycle_blue_chip_dip_pooled.decision_dates`'s own
verified bar-accounting (see that module's own docstring), just annual
cadence and a cross-sectional rather than per-ticker-absolute
condition.

Interpretation choices
-----------------------
* "Annual reformation following fiscal-year-end disclosures" is
  implemented as reformation on the last trading day of each CALENDAR
  YEAR present in the pool's own trading calendar (annual sibling of
  `capex_cycle_blue_chip_dip_pooled.month_end_mask` / `retail_inventory_
  sales_breadth_short.quarter_end_mask`) -- NOT Sloan's own May/June
  academic-replication reformation date. This is a real, disclosed
  choice: because entry/exit both read the POINT-IN-TIME (filed-date-
  gated) accrual ratio rather than a fixed-lag proxy, the specific
  calendar month chosen for reformation does not reintroduce a
  lookahead hazard either way -- it only changes which trading days are
  decision bars, and year-end matches this pipeline's own existing
  "reform at [period]-end" convention.
* Ranking closes over every ticker in `fundamentals_map` (the WHOLE
  declared pool, not just the calling ticker's own data) -- matches
  `abi1_driver.build_pooled_curve`'s own documented precedent for a
  cross-sectional pooled signal ("closing over every member's own price
  data to rank one ticker against the rest"; here it closes over every
  member's own FUNDAMENTALS instead of price, since ranking is on
  accrual, not price).
* A ticker missing from `fundamentals_map` (no usable EDGAR coverage,
  see `mining.accrual_coverage`) never enters the ranking at all --
  it can never be selected, but its absence does not lower
  `min_rank_members`'s bar for everyone else (the bar is n_defined
  among the tickers PRESENT in `fundamentals_map`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult
from mining.batch5.capex_cycle_blue_chip_dip_pooled import TRADED as _CAPEX_WATCHLIST_TRADED
from mining.edgar_point_in_time import point_in_time_asof

FAMILY_ID = "accrual_anomaly_sloan"
SHAPE = "pooled_single_asset"
BENCHMARK = "SPY"
# capex_cycle_blue_chip_dip_pooled's CIK-resolved watchlist (see the module
# docstring's "Universe" paragraph).
TRADED = list(_CAPEX_WATCHLIST_TRADED)
SIGNAL: list = []
SNAPSHOTS: dict = {}
PARAMETERS = {
    "signal": "total_accruals_to_total_assets",
    "accrual_formula": "(NetIncomeLoss_annual - CFO_annual) / average_total_assets",
    "rank_decile": "bottom_accrual",
    "bottom_fraction": 0.20,
    "min_rank_members": 10,
    "rebalance": "annual",
    "decision_frequency": "annual_at_calendar_year_end_close",
    "entry": "next_session_open",
    "exit": "same_reformation_bar_close_when_no_longer_bottom_ranked",
    "fundamentals_source": "edgar_point_in_time_filed_date_gated (NetIncomeLoss/"
                           "CFO annual duration, Assets instant)",
    "universe_source": (
        "PENDING_CONFIRMATION -- the same CIK-resolved 25-ticker watchlist "
        "as capex_cycle_blue_chip_dip_pooled (built by "
        "scripts/build_capex_watchlist.py), which also avoids overlap with "
        "retail_inventory_sales_breadth_short's own inventory-driven "
        "mechanism"
    ),
    "pooling": "all_pool_trades_concatenated_one_trial",
}
EXPECTED_MIN_TRADES = 0  # no minimum trade count is asserted for this family

BOTTOM_FRACTION = float(PARAMETERS["bottom_fraction"])
MIN_RANK_MEMBERS = int(PARAMETERS["min_rank_members"])

_ENTRY = SignalResult(entry=True, exit=False)
_EXIT = SignalResult(entry=False, exit=True)
_NONE = SignalResult()


def _normalize_index(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_localize(None)
    return out.normalize()


def year_end_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """True on the last bar of each calendar year present in ``index``
    (annual sibling of ``capex_cycle_blue_chip_dip_pooled.month_end_mask``
    / ``retail_inventory_sales_breadth_short.quarter_end_mask``)."""
    index = _normalize_index(index)
    periods = index.to_period("Y")
    mask = np.ones(len(index), dtype=bool)
    if len(index) > 1:
        mask[:-1] = periods[1:] != periods[:-1]
    return mask


def accrual_ratio_at(fundamentals: "dict | None", dates: pd.DatetimeIndex) -> pd.Series:
    """One company's point-in-time accrual-to-average-assets ratio at
    ``dates``. ``fundamentals`` = ``{"ni": df, "cfo": df, "assets": df}``,
    the three raw point-in-time frames ``mining.accrual_coverage.
    check_ticker_coverage`` produces for a coverage-ok ticker (NI/CFO from
    ``edgar_point_in_time.build_point_in_time_annual_series``, Assets from
    ``retail_inventory_coverage.build_point_in_time_instant_series``).
    Each leg is joined INDEPENDENTLY via ``point_in_time_asof`` (same
    "independently gated, not forced to share one fiscal year" convention
    as ``retail_inventory_sales_breadth_short._company_defined_and_ok``).
    NaN wherever any leg is undefined (missing/stale) or average assets
    is exactly zero. Returns ``NaN`` at every date if ``fundamentals`` is
    ``None`` or any leg is missing entirely -- graceful degradation,
    matching every sibling EDGAR-driven module."""
    dates = _normalize_index(dates)
    if not fundamentals:
        return pd.Series(np.nan, index=dates)
    ni, cfo, assets = (fundamentals.get("ni"), fundamentals.get("cfo"),
                       fundamentals.get("assets"))
    if (ni is None or ni.empty or cfo is None or cfo.empty
            or assets is None or assets.empty):
        return pd.Series(np.nan, index=dates)

    ni_joined = point_in_time_asof(ni, dates)["latest_val"]
    cfo_joined = point_in_time_asof(cfo, dates)["latest_val"]
    assets_joined = point_in_time_asof(assets, dates)
    latest_assets = assets_joined["latest_val"]
    prior_assets = assets_joined["prior_val"]
    # Sloan's own "average total assets" convention (mean of beginning and
    # ending assets) when a prior fiscal year is known; falls back to the
    # latest known assets alone otherwise (see module docstring).
    avg_assets = latest_assets.where(
        prior_assets.isna(), (latest_assets + prior_assets) / 2.0)

    defined = (ni_joined.notna() & cfo_joined.notna()
              & avg_assets.notna() & (avg_assets != 0))
    accrual = (ni_joined - cfo_joined) / avg_assets
    return accrual.where(defined, np.nan)


def selection_panel(dates: pd.DatetimeIndex, fundamentals_map: dict, *,
                    bottom_fraction: float = BOTTOM_FRACTION,
                    min_rank_members: int = MIN_RANK_MEMBERS) -> pd.DataFrame:
    """Bool DataFrame (``dates`` x tickers in ``fundamentals_map``) --
    True where that ticker is in the BOTTOM ``bottom_fraction`` of the
    pool's own accrual ratio at that date (lowest accrual = highest
    cash-flow quality, the LONG side of Sloan's anomaly), among tickers
    with a DEFINED accrual ratio that date. A date where fewer than
    ``min_rank_members`` tickers have a defined ratio has EVERY ticker
    False that date -- UNDEFINED ranking, never a spurious inclusion from
    a too-thin cross-section (same ``min_breadth_members`` convention as
    ``retail_inventory_sales_breadth_short.breadth_series``). Ties are
    broken alphabetically by ticker (deterministic, via a stable sort on
    columns already built in sorted order)."""
    dates = _normalize_index(dates)
    tickers = sorted(fundamentals_map)
    if not tickers or len(dates) == 0:
        return pd.DataFrame(False, index=dates, columns=tickers)

    ratios = pd.DataFrame(
        {ticker: accrual_ratio_at(fundamentals_map[ticker], dates) for ticker in tickers},
        index=dates)
    selected = pd.DataFrame(False, index=dates, columns=tickers)
    for date in dates:
        row = ratios.loc[date]
        defined_row = row.dropna()
        if len(defined_row) < min_rank_members:
            continue
        n_select = max(1, round(bottom_fraction * len(defined_row)))
        bottom = defined_row.sort_values(kind="stable").index[:n_select]
        selected.loc[date, bottom] = True
    return selected


def decision_dates(ticker: str, own_dates: pd.DatetimeIndex, fundamentals_map: dict, *,
                   bottom_fraction: float = BOTTOM_FRACTION,
                   min_rank_members: int = MIN_RANK_MEMBERS):
    """Pure sequential walk (mirrors ``capex_cycle_blue_chip_dip_pooled.
    decision_dates``'s own verified bar-accounting convention, minus its
    time-stop -- see module docstring): returns ``(entry_decision_bars,
    exit_bars)``, 0-indexed bar positions into ``own_dates``, paired 1:1.

    Reformation happens on the last bar of each calendar year present in
    ``own_dates`` (``year_end_mask``). While FLAT, the first reformation
    bar where ``ticker`` is selected (``selection_panel``, ranked against
    every ticker in ``fundamentals_map``) fires an entry, filling next-
    open. While LONG, the position is re-evaluated at each LATER
    reformation bar: the first where ``ticker`` is NOT selected fires an
    exit, filling at that SAME bar's close; while still selected the
    position rolls forward with no interim exit. A position still
    selected at the end of available data is a real, legitimate open
    position, censored at data end (no time-stop, unlike the sibling
    capex family) -- ``entries`` keeps that final entry with no matching
    ``exits`` element, and ``backtest_engine.run_backtest``'s own
    ``end_of_data`` forced-close handles it."""
    n = len(own_dates)
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    own_dates = _normalize_index(own_dates)
    reformation = year_end_mask(own_dates)

    reformation_dates = own_dates[reformation]
    panel = selection_panel(
        reformation_dates, fundamentals_map,
        bottom_fraction=bottom_fraction, min_rank_members=min_rank_members)
    selected_at_reformation = (panel[ticker] if ticker in panel.columns
                               else pd.Series(False, index=reformation_dates))
    selected_full = pd.Series(False, index=own_dates)
    selected_full.loc[selected_at_reformation.index] = selected_at_reformation.to_numpy()
    selected_ok = selected_full.to_numpy()

    entries: list = []
    exits: list = []
    i = 0
    while i < n:
        if not (reformation[i] and selected_ok[i]):
            i += 1
            continue
        decision_bar = i
        fill_bar = i + 1
        if fill_bar >= n:
            break
        entries.append(decision_bar)

        exit_bar = None
        j = fill_bar + 1
        while j < n:
            if reformation[j] and not selected_ok[j]:
                exit_bar = j
                break
            j += 1
        if exit_bar is None:
            # Still selected at the end of available data -- keep the
            # entry, emit no exit for it (see docstring), and stop.
            break
        exits.append(exit_bar)
        i = exit_bar

    return np.array(entries, dtype=np.int64), np.array(exits, dtype=np.int64)


def _make_signal_fn(ticker: str, index, frames: dict, fundamentals_map: "dict | None") -> callable:
    if ticker not in frames:
        raise KeyError(f"missing frame for {ticker}")
    frame = frames[ticker]
    if "Close" not in frame.columns:
        raise ValueError(f"{ticker} frame must have a Close column")

    bucket = _normalize_index(index)
    if len(bucket) == 0:
        def empty_signal_fn(frame_slice, i):
            return _NONE
        return empty_signal_fn
    cutoff = bucket[-1]

    idx = _normalize_index(frame.index)
    own = frame.set_axis(idx)
    own = own.loc[own.index <= cutoff]
    if not own.index.equals(bucket):
        raise ValueError(f"{FAMILY_ID}: bucket must equal the traded frame's "
                         f"own calendar truncated to its last bar")

    entries, exit_bars = decision_dates(ticker, bucket, fundamentals_map or {})

    entry_mask = np.zeros(len(bucket), dtype=bool)
    exit_mask = np.zeros(len(bucket), dtype=bool)
    entry_mask[entries] = True
    exit_mask[exit_bars] = True
    n = len(bucket)
    bucket_ns = bucket.asi8
    by_date = {int(ts): pos for pos, ts in enumerate(bucket_ns)}

    def signal_fn(frame_slice, i):
        today = pd.Timestamp(frame_slice.index[-1])
        if today.tzinfo is not None:
            today = today.tz_localize(None)
        today_ns = int(today.normalize().value)
        pos = i if (0 <= i < n and bucket_ns[i] == today_ns) else by_date.get(today_ns)
        if pos is None:
            return _NONE
        entry = bool(entry_mask[pos])
        exit_ = bool(exit_mask[pos])
        if entry and exit_:
            return SignalResult(entry=True, exit=True)
        if entry:
            return _ENTRY
        if exit_:
            return _EXIT
        return _NONE

    return signal_fn


def make_signal_fn(ticker: str, index, frames: dict, *, fundamentals_map=None):
    """SignalFn factory matching every sibling module's ``(ticker, index,
    frames)`` contract, with an OPTIONAL keyword-only ``fundamentals_map``
    (``{ticker: {"ni": df, "cfo": df, "assets": df}}`` covering the WHOLE
    pool, not just this ticker -- the cross-sectional ranking this
    family's mechanism needs closes over every pool member's own
    fundamentals, matching ``abi1_driver.build_pooled_curve``'s own
    documented "closing over every member's own data to rank one ticker
    against the rest" precedent). Default (``None``) keeps every
    reformation date UNDEFINED for this ticker (never enters), same
    graceful-degradation convention as every sibling pooled EDGAR
    module."""
    return _make_signal_fn(ticker, index, frames, fundamentals_map)
