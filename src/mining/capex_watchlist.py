"""Pure ranking core of the capex-cycle blue-chip dip-buy mechanical watchlist rule
("the N largest
S&P 500 constituents by trailing capex spend as of a fixed pre-2020
date") -- an objective, mechanical selection, not a hand-picked list.

Kept separate from the live-fetch orchestration
(``scripts/build_capex_watchlist.py``) the same way ``edgar_data.py``
and ``edgar_point_in_time.py`` split raw fetch from pure logic -- this
module takes already-built point-in-time series (the output shape of
``edgar_point_in_time.build_point_in_time_annual_series``) and does the
as-of lookup + ranking, none of it touching the network.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from mining.edgar_data import REQUIRED_FACT_FIELDS
from mining.edgar_point_in_time import STALENESS_CAP_DAYS, point_in_time_asof


def trailing_capex_asof(series: Optional[pd.DataFrame], asof_date,
                        *, staleness_cap_days: int = STALENESS_CAP_DAYS) -> Optional[float]:
    """The trailing annual capex figure known as of ``asof_date`` for one
    company's point-in-time series (``build_point_in_time_annual_series``'s
    output shape). Returns ``None`` -- never NaN -- when the series is
    missing/empty, the date predates the company's first usable
    disclosure, or the nearest disclosure is stale beyond
    ``staleness_cap_days`` (``point_in_time_asof``'s own guard)."""
    if series is None or len(series) == 0:
        return None
    joined = point_in_time_asof(
        series, pd.DatetimeIndex([pd.Timestamp(asof_date)]),
        staleness_cap_days=staleness_cap_days)
    value = joined["latest_val"].iloc[0]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def rank_by_trailing_capex(capex_by_ticker: dict, n: int) -> list:
    """Rank tickers by trailing capex descending, dropping any ``None``/
    NaN value (a ticker with no usable capex disclosure as of the fixed
    date is excluded, never treated as zero), and take the top ``n``.
    Ties break alphabetically by ticker so the result is deterministic
    and reproducible from the same input on a re-run. Returns a list of
    ``{"ticker": str, "trailing_capex_usd": float}``, ranked descending."""
    usable = [
        (ticker, float(value)) for ticker, value in capex_by_ticker.items()
        if value is not None and not (isinstance(value, float) and math.isnan(value))
    ]
    usable.sort(key=lambda pair: (-pair[1], pair[0]))
    return [{"ticker": ticker, "trailing_capex_usd": value}
            for ticker, value in usable[:n]]


def capex_growth_multiple(base_val: Optional[float], recent_val: Optional[float],
                          *, min_base_val: float = 0.0) -> Optional[float]:
    """``recent_val / base_val``, or ``None`` if either value is missing,
    ``base_val`` is non-positive (a zero/negative baseline makes the
    ratio undefined or economically meaningless), or ``base_val`` is
    below ``min_base_val``. The watchlist's thesis
    is "huge capex for AI/infrastructure that will pay off big, with
    hyperscalers currently being punished for it" -- ranking by GROWTH
    (this function) rather than raw level is what targets that, since a
    company's ABSOLUTE capex level as of an old fixed date (the level
    ranking ``rank_by_trailing_capex`` produces) can completely miss a
    company whose spending only ramped up later. Example: a company whose
    capex rose many-fold from FY2019 on an AI/cloud infrastructure buildout
    is invisible to a level-as-of-2019 ranking.

    ``min_base_val`` (default 0.0, i.e. no floor) guards against a
    genuine EDGAR data hazard: American Electric
    Power's FY2018 capex was disclosed as 14.6 million USD in the filing made
    before a 2019-12-31 base date, then RESTATED to exactly zero for the
    identical period in the next year's comparative disclosure -- a
    provable filer inconsistency (confirmed by the later restatement,
    not merely suspected), not a genuine tiny base. A growth RATIO,
    unlike the level ranking, is extremely
    sensitive to a spuriously small denominator -- dividing a normal
    recent figure by 14.6 million USD manufactures a 236x "growth" reflecting no
    real acceleration. Verified this is NOT the same as a company having
    a genuinely modest-but-real base (Ventas/VTR's 131.9 million USD base is
    consistent across multiple filings, no restatement anomaly, and a
    real ~22x increase in its capital deployment)."""
    if base_val is None or recent_val is None:
        return None
    if base_val <= 0:
        return None
    if base_val < min_base_val:
        return None
    return recent_val / base_val


def rank_by_capex_growth(level_by_ticker: dict, growth_by_ticker: dict, *,
                         level_floor_rank: int, n: int) -> list:
    """Rank tickers by capex GROWTH (``capex_growth_multiple``) descending,
    restricted to tickers whose RECENT absolute level (``level_by_ticker``)
    places them in the top ``level_floor_rank`` by level. The floor exists
    so a tiny company's huge percentage swing off a trivial base cannot
    dominate a pure growth-rate ranking (e.g. a company going from 1 million to
    50 million USD of capex is "5000% growth" but economically irrelevant) -- only
    genuinely large-scale capital programs that are ALSO accelerating
    qualify. A ticker missing from ``level_by_ticker``, below the floor,
    or with an undefined growth value (``None``) is excluded. Ties break
    alphabetically by ticker for determinism. Returns a list of
    ``{"ticker": str, "capex_growth_multiple": float, "recent_capex_usd":
    float}``, ranked descending by growth."""
    ranked_by_level = sorted(
        ((ticker, value) for ticker, value in level_by_ticker.items()
         if value is not None and not (isinstance(value, float) and math.isnan(value))),
        key=lambda pair: (-pair[1], pair[0]))
    eligible = {ticker for ticker, _ in ranked_by_level[:level_floor_rank]}

    usable = [
        (ticker, float(growth_by_ticker[ticker]), float(level_by_ticker[ticker]))
        for ticker in eligible
        if growth_by_ticker.get(ticker) is not None
        and not (isinstance(growth_by_ticker[ticker], float) and math.isnan(growth_by_ticker[ticker]))
    ]
    usable.sort(key=lambda row: (-row[1], row[0]))
    return [{"ticker": ticker, "capex_growth_multiple": growth, "recent_capex_usd": level}
            for ticker, growth, level in usable[:n]]


def dedupe_share_classes_by_cik(capex_by_ticker: dict, cik_by_ticker: dict) -> dict:
    """Drop every ticker but one per CIK. For example, GOOG and
    GOOGL share one CIK (Alphabet) and would otherwise occupy 2 of a
    pooled family's N slots as if they were two independent companies,
    inflating within-pool correlation and silently shrinking the real
    economic breadth of an "N distinct companies" selection rule to
    N-1. Keeps the alphabetically-first ticker for each CIK --
    deterministic and reproducible on a re-run, not a judgment call
    about which share class is "more representative." A ticker with no
    resolved CIK (``None``) is never treated as colliding with another
    unresolved ticker -- only a REAL shared CIK collides."""
    kept_ticker_by_cik: dict = {}
    for ticker in sorted(capex_by_ticker):
        cik = cik_by_ticker.get(ticker)
        if cik is None:
            kept_ticker_by_cik[("unresolved", ticker)] = ticker
            continue
        existing = kept_ticker_by_cik.get(cik)
        if existing is None or ticker < existing:
            kept_ticker_by_cik[cik] = ticker
    kept_tickers = set(kept_ticker_by_cik.values())
    return {ticker: value for ticker, value in capex_by_ticker.items()
            if ticker in kept_tickers}


def resolve_tag_conflicts(facts_by_tag: dict, tag_priority) -> pd.DataFrame:
    """Combine multiple XBRL tags' raw fact frames (``edgar_data.fetch_
    all_tags_for_company``'s ``{tag: DataFrame}`` output) into one frame,
    resolving any exact ``(start, end, filed, fp, form)`` combination
    disclosed under more than one tag by preferring the LARGER value,
    never by fetch or dict/concat order.

    Example (Southern Co, CIK 92122): a company can
    genuinely co-report the SAME period on the SAME filing date under
    two DIFFERENT tags measuring related but distinct things (e.g. plain
    PP&E purchases vs. PP&E purchases + capitalized interest) -- the
    multi-tag fallback list's own "a company uses tag A until it
    switches to tag B" assumption does not always hold. Without this,
    ``pd.concat`` + ``edgar_point_in_time.build_point_in_time_annual_
    series``'s unconditional last-write-wins fold picks whichever tag
    happened to be fetched/concatenated last -- accidental, not a
    defensible rule. Every period NOT in conflict (the intended
    different-tag-per-era case) is preserved from whichever tag reported
    it, untouched.

    Why the larger value, not ``tag_priority``'s listed order
    (``edgar_data.CAPEX_TAGS``, narrowest/most-standard tag
    first): the narrow tag can be actively WRONG
    for Devon Energy (DVN, CIK 1090012): its "narrowest" tag,
    PaymentsToAcquirePropertyPlantAndEquipment, captured a nearly
    irrelevant ~31-55 million USD/yr subset while PaymentsToAcquireProductiveAssets
    captured its real ~1.9-2.5 billion USD/yr oil & gas development spend for the
    SAME fiscal years -- "narrowest tag" and "most representative tag"
    are not the same thing, and preferring the narrow one would produce a
    65x false "growth" figure once DVN stopped dual-reporting and only
    the wide tag remained. Preferring the larger value is more robust
    (undercounting real capex is the worse failure mode here) and still
    gets Southern Co right (PP&E+capitalized-interest, the larger
    figure, is also the more economically complete one). ``tag_priority``
    is used only as a deterministic tie-break for a genuine EXACT
    value tie across tags (rare) -- kept as a parameter rather than
    removed so callers don't need two separate calling conventions.

    Stability first: "larger" is not always "more representative" either --
    Ventas (VTR, CIK 740260)'s PaymentsToAcquireProductiveAssets
    reported FY2017 capex as 811.875 million USD when first filed, then RESTATED
    the SAME period to 1.096 billion USD a year later (a ~35% revision), while its
    PaymentsToAcquirePropertyPlantAndEquipment reported the same FY2017
    figure identically both times. A tag with a KNOWN restatement in ITS
    OWN recent history is a weaker source than one with none, so
    STABILITY outranks size: a tag is "unstable" for a given
    conflict row if any TWO of its own facts covering a period within
    ``STABILITY_LOOKBACK_YEARS`` (3) of that row's own period, both filed
    ON OR BEFORE that row's own ``filed`` date, disagree on value for the
    same ``(start, end)`` -- using only what was knowable at that row's
    own filed date, never a later restatement (no lookahead), and
    bounded to RECENT history only: an unbounded "any restatement, ever"
    check would flag VTR's CORRECT (stable) tag as unstable because of an
    unrelated restatement from 2010-2012 (the early-XBRL-adoption era),
    nearly a decade before the conflict being resolved. If exactly one side of a conflict is unstable by this
    test, the STABLE side wins regardless of magnitude; otherwise (both
    stable, or both unstable) the larger-value rule above decides,
    unchanged.

    ``fp``/``form`` are part of the conflict key: SEC XBRL routinely reports the
    IDENTICAL ``(start, end, filed)`` triple twice under one tag, once
    ``fp="FY"``/``form="10-K"`` (the real annual fact) and once
    ``fp="Q4"``/``form="10-Q"`` (a duration-cumulative duplicate, not
    annual-qualifying). Keying on ``(start, end, filed)`` alone would
    treat these as a same-tag "conflict" and silently keep whichever one
    iterated first -- capable of discarding the real annual fact with no
    error. Including ``fp``/``form`` means these never collide; only a
    genuine same-context (same period, same filing, same fp/form) value
    disagreement across DIFFERENT tags is resolved."""
    columns = list(REQUIRED_FACT_FIELDS)
    if not facts_by_tag:
        return pd.DataFrame(columns=columns)

    STABILITY_LOOKBACK_YEARS = 3  # ignore restatements older than this,
    # relative to the period being resolved -- for example
    # (Ventas/VTR, CIK 740260): an UNSCOPED "any restatement, ever" check
    # would flag the CORRECT tag unstable because of an old restatement
    # from 2010-2012 (the notoriously messy early-XBRL-adoption era)
    # nearly a decade before the conflict being resolved, and silently
    # fall back to "prefer larger" for that conflict. Only RECENT
    # reliability is a meaningful signal.

    def is_unstable_near(frame: pd.DataFrame, as_of_filed, ref_end) -> bool:
        known = frame.loc[frame["filed"] <= as_of_filed]
        cutoff = ref_end - pd.DateOffset(years=STABILITY_LOOKBACK_YEARS)
        known = known.loc[known["end"] >= cutoff]
        # Group by (start, end, fp, form), matching the main resolution
        # key's granularity: grouping by (start, end) alone could conflate two
        # GENUINELY DIFFERENT disclosure contexts reported under the
        # SAME tag (e.g. an FY/10-K annual fact and a Q4/10-Q duration-
        # cumulative one for the same period, the routine SEC pattern the
        # fp/form part of the main conflict key handles) as if the tag were
        # disagreeing with ITSELF.
        # A different fp/form is a different disclosure, not a revision.
        for _, group in known.groupby(["start", "end", "fp", "form"]):
            if group["val"].nunique() > 1:
                return True
        return False

    tag_rank = {tag: rank for rank, tag in enumerate(tag_priority)}
    # (start, end, filed, fp, form) -> (val, is_unstable, priority_rank, row_dict)
    best: dict = {}
    for tag, frame in facts_by_tag.items():
        rank = tag_rank.get(tag, len(tag_priority))
        for _, row in frame.iterrows():
            key = (row["start"], row["end"], row["filed"], row["fp"], row["form"])
            val = row["val"]
            unstable = is_unstable_near(frame, row["filed"], row["end"])
            existing = best.get(key)
            if existing is None:
                best[key] = (val, unstable, rank, row.to_dict())
                continue
            existing_val, existing_unstable, existing_rank, _ = existing
            if unstable != existing_unstable:
                # Stability outranks size: the stable side always wins.
                if existing_unstable:
                    best[key] = (val, unstable, rank, row.to_dict())
                continue
            if val > existing_val or (val == existing_val and rank < existing_rank):
                best[key] = (val, unstable, rank, row.to_dict())

    combined = pd.DataFrame([entry[3] for entry in best.values()], columns=columns)
    return combined.sort_values("filed", kind="stable").reset_index(drop=True)
