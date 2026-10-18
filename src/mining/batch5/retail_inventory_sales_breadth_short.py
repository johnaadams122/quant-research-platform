"""Candidate: retail_inventory_sales_breadth_short (shape panel_breadth).

Mechanism: using SEC EDGAR XBRL inventory (``InventoryNet`` and its real
retail-specific siblings, see ``mining.edgar_data.INVENTORY_TAGS``) and
sales/revenue tags (``mining.edgar_data.REVENUE_TAGS``) for a liquid
retail-sector universe, SHORT XRT (SPDR S&P Retail ETF) for the following
quarter when sector-wide breadth of retailers showing inventory growth
outpacing sales growth crosses a threshold. Economic rationale:
documented inventory-driven margin-compression anomaly (Bernard & Thomas
1989) -- a retailer whose inventory is building faster than its sales
is heading toward markdowns/margin compression; measuring this across a
whole sector's breadth (rather than one name) is a distinct signal from
the price/momentum sector families and from the capex-based EDGAR
families in this pipeline (a different fundamental axis
entirely -- inventory/sales divergence, not capital-spending growth).

Universe: XRT's own current holdings (a dated snapshot of the fund
sponsor's daily holdings file) -- the correct, non-circular population for a
"breadth of the sector this ETF tracks" signal, same "the screened
population IS the traded population" discipline as
``capex_cycle_blue_chip_dip_pooled.py``'s own watchlist. Coverage
check (``mining.retail_inventory_coverage``, a disclosed real risk of this
candidate's design): names without a usable, point-in-time inventory AND
revenue series are excluded -- typically marketplace/platform businesses
with no traditional retail inventory, a too-sparse primary tag (an
inventory tag used only in a note disclosure rather than the balance-sheet
line), an unrelated business model, or a transient EDGAR fetch failure.

RULE
----
    breadth_threshold = 0.50   # fraction of the coverage-checked universe
    min_breadth_members = 20   # floor before breadth is even defined
    at each quarter-end bar m:
      for each company C in the coverage-checked universe:
        inv_growth = latest known inventory YoY growth (point-in-time,
                     filed-date/EDGAR-disclosure gated, from its INSTANT
                     balance-sheet tag)
        sales_growth = latest known revenue YoY growth (point-in-time,
                       same gating, from its DURATION revenue tag)
        defined = both inv_growth and sales_growth are known and not
                  stale (edgar_point_in_time.STALENESS_CAP_DAYS)
        ok = defined AND inv_growth > sales_growth
      breadth = (# companies with ok) / (# companies with defined),
                or UNDEFINED if fewer than min_breadth_members companies
                are defined at this date (never treated as breadth=0)
      if breadth >= breadth_threshold and position==FLAT:
        SHORT XRT at the OPEN of the next trading session
      if position==SHORT and (breadth is UNDEFINED or breadth <
        breadth_threshold at the NEXT quarter-end): COVER at that
        quarter-end's close
      if position==SHORT and breadth remains >= breadth_threshold at the
        next quarter-end: ROLL FORWARD (no interim exit) -- "for the
        following quarter" is a re-evaluated commitment, not a fixed
        one-shot hold, matching this repo's own quarterly-rebalance
        convention elsewhere in the pipeline
      one trade = one contiguous SHORT holding period

DISCLOSED DEFAULTS (this repo's standing disclosed-default convention):
``breadth_threshold`` (a plain majority reading of "crosses a threshold")
and ``min_breadth_members`` (a floor loosely analogous to
``panel.py``'s own ``min_members`` convention for its 11-member sector
panel, scaled to this candidate's ~68-member universe) are both proposed
defaults chosen by reasoning about the rule, not values fitted to or
validated against data.

Engine mapping (mining.backtest_engine.run_backtest): direction="short"
is the DRIVER's job (this module's own SignalResult never encodes
direction, matching the engine's own separation of signal from
direction) -- entries/exits are precomputed once per bucket exactly like
every sibling single-instrument module's own walk, then read back by
signal_fn.

Interpretation choices
-----------------------
* "Quarter-end" = the last bar of each CALENDAR quarter present in the
  traded frame's own index (mirrors month_end_mask's identical
  convention in the sibling capex family, just quarterly).
* A company missing EITHER leg (inventory or revenue) at a given date is
  EXCLUDED from that date's breadth entirely -- never counted as "not
  ok". This is what ``mining.retail_inventory_coverage``'s own coverage
  check exists to bound: only the coverage-checked subset of the
  holdings ever contributes a usable pair at all, and even among those,
  EDGAR disclosure timing
  means not every one is defined on every historical quarter-end (a
  company's fundamentals are only known once its first usable filing
  clears the pipeline).
* Both legs' point-in-time joins reuse ``edgar_point_in_time.
  point_in_time_asof`` UNCHANGED -- it is fully generic over which value
  column a series carries, so it applies identically to the DURATION
  revenue series (``capex_growth_yoy``) and the INSTANT inventory series
  (``growth_yoy``) this candidate's own coverage-check module produces.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mining.backtest_engine import SignalResult
from mining.edgar_point_in_time import point_in_time_asof

FAMILY_ID = "retail_inventory_sales_breadth_short"
SHAPE = "panel_breadth"
TRADED = ["XRT"]
SIGNAL: list = []  # external (EDGAR inventory/revenue fundamentals panel
                   # across XRT's own real holdings), not a price-frame ticker
BENCHMARK = "SPY"
SNAPSHOTS: dict = {}
PARAMETERS = {
    "breadth_threshold": 0.50,
    "min_breadth_members": 20,
    "decision_frequency": "quarterly_at_quarter_end_close",
    "entry": "next_session_open",
    "exit": "next_quarter_end_close_if_breadth_drops_below_threshold",
    "hold": "roll_forward_while_still_triggered",
    "direction": "short",
    "universe_source": (
        "CONFIRMED: XRT's own current holdings (a dated snapshot of the fund "
        "sponsor's daily holdings file), coverage-checked by "
        "mining.retail_inventory_coverage, applied backward over the full "
        "backtest window. Disclosed survivorship-bias caveat: retailers "
        "delisted for distress (e.g. Sears, Toys R Us, Bed Bath & Beyond, "
        "RadioShack) are invisible to this proxy universe -- judged "
        "CONSERVATIVE (understates real crisis-period breadth rather than "
        "inflating it), same disclosed-caveat convention as "
        "money_market_fund_flow_riskoff's burn-in issue."
    ),
    "inventory_growth_source": "edgar_point_in_time_filed_date_gated_annual_instant",
    "sales_growth_source": "edgar_point_in_time_filed_date_gated_annual_duration",
}
EXPECTED_MIN_TRADES = 1  # Configured conservative floor (a rare-event
# breadth signal), same conservative-floor convention as
# fed_net_liquidity_regime.py.

BREADTH_THRESHOLD = PARAMETERS["breadth_threshold"]
MIN_BREADTH_MEMBERS = int(PARAMETERS["min_breadth_members"])

_ENTRY = SignalResult(entry=True, exit=False)
_EXIT = SignalResult(entry=False, exit=True)
_NONE = SignalResult()


def _normalize_index(index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(index))
    if out.tz is not None:
        out = out.tz_localize(None)
    return out.normalize()


def quarter_end_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """True on the last bar of each calendar quarter present in
    ``index`` (quarterly sibling of every month_end_mask helper
    elsewhere in this pipeline)."""
    index = _normalize_index(index)
    periods = index.to_period("Q")
    mask = np.ones(len(index), dtype=bool)
    if len(index) > 1:
        mask[:-1] = periods[1:] != periods[:-1]
    return mask


def _company_defined_and_ok(inventory: pd.DataFrame, revenue: pd.DataFrame,
                            dates: pd.DatetimeIndex) -> "tuple[np.ndarray, np.ndarray]":
    """For one company: ``(defined, ok)`` boolean arrays over ``dates``.
    ``defined`` requires BOTH legs to have a non-stale point-in-time
    value; ``ok`` (only meaningful where ``defined``) is inventory
    growth strictly exceeding sales growth."""
    if inventory is None or inventory.empty or revenue is None or revenue.empty:
        false = np.zeros(len(dates), dtype=bool)
        return false, false
    inv_joined = point_in_time_asof(inventory, dates)["growth_yoy"]
    rev_joined = point_in_time_asof(revenue, dates)["capex_growth_yoy"]
    defined = inv_joined.notna().to_numpy() & rev_joined.notna().to_numpy()
    ok = defined & (inv_joined.to_numpy() > rev_joined.to_numpy())
    return defined, ok


def breadth_series(dates: pd.DatetimeIndex, fundamentals_map: dict, *,
                   min_members: int = MIN_BREADTH_MEMBERS) -> pd.Series:
    """Sector-wide breadth at each date in ``dates``: the fraction of
    companies in ``fundamentals_map`` (``{ticker: {"inventory": df,
    "revenue": df}}``, each df the point-in-time series shape this
    candidate's own coverage-check module produces) whose inventory
    growth currently exceeds sales growth, among companies with BOTH
    legs currently defined. A date where fewer than ``min_members``
    companies are defined gets NaN, never a breadth computed from a
    too-thin, potentially biased subset -- this candidate's own
    disclosed coverage risk, carried through to the signal itself, not
    just the pre-flight check."""
    dates = _normalize_index(dates)
    n = len(dates)
    if not fundamentals_map:
        return pd.Series(np.nan, index=dates)

    ok_count = np.zeros(n, dtype=int)
    defined_count = np.zeros(n, dtype=int)
    for entry in fundamentals_map.values():
        defined, ok = _company_defined_and_ok(
            entry.get("inventory"), entry.get("revenue"), dates)
        defined_count += defined.astype(int)
        ok_count += ok.astype(int)

    with np.errstate(invalid="ignore", divide="ignore"):
        breadth = np.where(defined_count > 0, ok_count / np.maximum(defined_count, 1),
                           np.nan)
    breadth = np.where(defined_count >= min_members, breadth, np.nan)
    return pd.Series(breadth, index=dates)


def decision_dates(breadth_ok: np.ndarray, quarter_end: np.ndarray):
    """Pure sequential walk (mirrors the sibling capex family's own
    verified bar-accounting convention): returns ``(entry_decision_bars,
    exit_bars)``, 0-indexed bar positions, paired 1:1.

    While FLAT, the first quarter-end bar where ``breadth_ok`` holds
    fires an entry, filling next-open. While SHORT, the position is
    re-evaluated only at the NEXT quarter-end bar: if ``breadth_ok`` is
    still True there, NO exit fires (the short rolls forward, matching
    "for the following quarter" as a re-evaluated commitment); the first
    quarter-end where it is False fills the exit at that bar's close.
    After an exit, the walk resumes checking AT the exit bar itself
    (same chained-reentry convention as the sibling capex family).

    UNLIKE the sibling capex family (which bounds every hold with
    ``max_hold_sessions`` and discards an entry that cannot find an exit
    within the data), this mechanism has no time-stop: a position that
    stays triggered through the end of available data is a real,
    legitimate open position, censored at data end -- ``entries`` KEEPS
    that final entry with no matching ``exits`` element, and
    ``backtest_engine.run_backtest``'s own ``end_of_data`` forced-close
    handles it (no signal exit is ever emitted for it, matching every
    other single-instrument module's reliance on that same engine path)."""
    n = len(breadth_ok)
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    entries: list = []
    exits: list = []
    i = 0
    while i < n:
        if not (quarter_end[i] and breadth_ok[i]):
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
            if quarter_end[j] and not breadth_ok[j]:
                exit_bar = j
                break
            j += 1
        if exit_bar is None:
            # Still open at the end of available data -- keep the entry,
            # emit no exit for it (see docstring), and stop: there is no
            # more data to walk.
            break
        exits.append(exit_bar)
        i = exit_bar

    return np.array(entries, dtype=np.int64), np.array(exits, dtype=np.int64)


def _make_signal_fn(ticker: str, index, frames: dict, fundamentals_map: dict,
                    *, min_members: int, threshold: float) -> callable:
    if ticker != TRADED[0]:
        raise ValueError(f"{FAMILY_ID}: make_signal_fn expects ticker "
                         f"{TRADED[0]!r}, got {ticker!r}")
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

    breadth = breadth_series(bucket, fundamentals_map or {}, min_members=min_members)
    breadth_ok = (breadth >= threshold).fillna(False).to_numpy()
    qe = quarter_end_mask(bucket)
    entries, exit_bars = decision_dates(breadth_ok, qe)

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


def make_signal_fn(ticker: str, index, frames: dict, *, fundamentals_map=None,
                   min_members: int = MIN_BREADTH_MEMBERS,
                   threshold: float = BREADTH_THRESHOLD):
    """SignalFn factory matching every sibling module's ``(ticker, index,
    frames)`` contract, with an OPTIONAL keyword-only ``fundamentals_map``
    (``{ticker: {"inventory": df, "revenue": df}}``) -- this family's own
    bespoke driver always passes it explicitly; the default (``None`` ->
    breadth is UNDEFINED everywhere -> never enters) keeps
    ``validate_module_contract``'s generic 3-positional-arg expectation
    satisfied without a shared-driver signature change anywhere in this
    repo, matching ``capex_cycle_blue_chip_dip_pooled.make_signal_fn``'s
    own precedent exactly. Direction ("short") is NOT encoded here -- the
    driver passes ``direction="short"`` to ``backtest_engine.run_backtest``
    directly, matching that engine's own separation of signal from
    direction."""
    return _make_signal_fn(
        ticker, index, frames, fundamentals_map,
        min_members=min_members, threshold=threshold)
