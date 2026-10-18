"""Builds the frozen capex-cycle blue-chip dip-buy watchlist
via an objective, mechanical selection rule: an objective rule, not a
hand-picked list.

**Why growth, not level.** A simpler design would rank by trailing capex LEVEL as of a fixed
pre-2020 date -- but the underlying thesis is "huge capex for AI and
other infrastructure that will pay off big, and hyperscalers currently
being punished for it in the markets." A level-as-of-2019 ranking misses
that almost entirely: Oracle's real capex went from about 1.66 billion USD
(FY2019) to 55.7 billion USD (FY2026), a ~33x increase driven by AI/cloud buildout, and was
nowhere near an N=25 cutoff. Selecting by CURRENT capex level alone
isn't the fix either -- it doesn't target "accelerating," just "big" (a
utility's routine grid capex would still crowd out a company whose capex
is exploding from a smaller base). So the rule ranks by capex GROWTH
(``mining.capex_watchlist.capex_growth_multiple``: most recent available
annual disclosure / disclosure as of BASE_DATE), gated by a floor on
absolute RECENT level (``LEVEL_FLOOR_RANK``: must be in the top N by
CURRENT capex to even be eligible for the growth ranking) so a tiny
company's huge percentage swing off a trivial base can't dominate. This
is fully mechanical and pre-registered BEFORE any dry run/backtest
uses this watchlist -- it does not use future STOCK returns to pick
names, only already-disclosed, point-in-time-correct capex figures (the
point-in-time join is filed-date-gated, never end-date).

Universe = a FRESH live fetch of just the S&P 500 table from Wikipedia
(not the existing data/mining/universe_snapshot_2026-08-29.json, which
merges in NASDAQ-100 and is not separable back into an S&P-500-only
subset) -- current constituents only, same disclosed survivorship-bias
convention every other pooled family in this repo already carries (see
large_cap_single_stock_pooled_tsmom.py's PARAMETERS["pool"] prose). No
point-in-time S&P 500 MEMBERSHIP history exists anywhere in this repo
-- only the CAPEX figures used for
ranking are point-in-time (via filed date, never end date); which
companies were even eligible to be ranked is today's list.

BASE_DATE (2019-12-31, pre-AI-supercycle anchor), LEVEL_FLOOR_RANK (100),
and N (25) are PROVISIONAL_DEFAULT values, same "ship with a
disclosed placeholder" convention as every other invented constant in
this pipeline. RECENT_DATE is always "today" (the run date) -- not a
placeholder, since "most recent available disclosure" is the whole
point.

Usage (real network calls -- Wikipedia + SEC; takes several minutes,
self-throttled to SEC's documented rate limit):
    python scripts/build_capex_watchlist.py
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from io import StringIO

import pandas as pd
import requests

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from mining import edgar_cik_resolution  # noqa: E402
from mining import sec_contact  # noqa: E402
from mining import capex_watchlist  # noqa: E402
from mining import universe  # noqa: E402
from mining.edgar_data import CAPEX_TAGS, fetch_all_tags_for_company  # noqa: E402
from mining.edgar_point_in_time import (  # noqa: E402
    annual_facts_only, build_point_in_time_annual_series)

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Same header convention as scripts/fetch_index_constituents.py -- Wikipedia
# blocks the default urllib UA with a 403.
_WIKIPEDIA_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                    "quant-research-platform-script/1.0"}

BASE_DATE = pd.Timestamp("2019-12-31")  # PROVISIONAL_DEFAULT
WATCHLIST_SIZE = 25  # PROVISIONAL_DEFAULT (target range: 20-30)
LEVEL_FLOOR_RANK = 100  # PROVISIONAL_DEFAULT -- must be top-100 by CURRENT
                        # absolute capex level to even be eligible for growth ranking
MIN_BASE_CAPEX_USD = 50_000_000  # PROVISIONAL_DEFAULT -- guards against a
                        # spuriously tiny base (one a filer later restates
                        # for the same period), which would manufacture an extreme,
                        # unreal growth ratio; see capex_growth_multiple's own docstring


def _fetch_sp500_tickers(http_get=requests.get) -> list:
    resp = http_get(SP500_URL, headers=_WIKIPEDIA_HEADERS, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    for t in tables:
        if "Symbol" in t.columns and len(t) > 50:
            return t["Symbol"].astype(str).tolist()
    raise ValueError(f"build_capex_watchlist: no S&P 500 table found at {SP500_URL} "
                      f"with a 'Symbol' column and >50 rows -- page structure may "
                      f"have changed")


def _fetch_company_tickers(http_get=requests.get) -> dict:
    resp = http_get(COMPANY_TICKERS_URL,
                    headers={"User-Agent": sec_contact.sec_contact_user_agent()},
                    timeout=30)
    resp.raise_for_status()
    return resp.json()


def _build_point_in_time_series_for_ticker(cik: int, *, user_agent: str | None = None,
                                            sleep=time.sleep, min_interval: float):
    """Fetch every CAPEX_TAGS variant for one company and fold whichever
    ones actually returned data into a single point-in-time annual
    series. Returns (series_or_None, tags_used: list[str])."""
    by_tag = fetch_all_tags_for_company(
        cik, CAPEX_TAGS, user_agent=user_agent, sleep=sleep, min_interval=min_interval)
    if not by_tag:
        return None, []
    # Filter each tag to annual-only BEFORE combining, so resolve_tag_conflicts
    # only ever sees genuine FY/10-K rows, never a same-period Q4/10-Q
    # duplicate that could otherwise collide with them (its own key
    # includes fp/form as a second line of defense, but filtering first
    # means that collision can never even be attempted here).
    annual_by_tag = {tag: annual_facts_only(frame) for tag, frame in by_tag.items()}
    annual_by_tag = {tag: frame for tag, frame in annual_by_tag.items() if not frame.empty}
    if not annual_by_tag:
        return None, list(by_tag.keys())
    # resolve_tag_conflicts, not a blind pd.concat: a company can
    # genuinely co-report the SAME period under two different tags with
    # DIFFERENT values (e.g. Southern Co, CIK 92122) -- the LARGER value
    # decides, not tag-list order (see resolve_tag_conflicts's own docstring), never fetch/concat order.
    annual = capex_watchlist.resolve_tag_conflicts(annual_by_tag, tag_priority=CAPEX_TAGS)
    if annual.empty:
        return None, list(by_tag.keys())
    series = build_point_in_time_annual_series(annual)
    return series, list(by_tag.keys())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=os.path.join(_ROOT, "data", "mining"))
    parser.add_argument("--date", default=None,
                        help="Snapshot-date stamp for the output filename "
                             "(default: today, UTC date).")
    parser.add_argument("--n", type=int, default=WATCHLIST_SIZE)
    parser.add_argument("--base-asof", default=str(BASE_DATE.date()),
                        help="Growth-baseline date (pre-AI-supercycle anchor).")
    parser.add_argument("--level-floor-rank", type=int, default=LEVEL_FLOOR_RANK,
                        help="A ticker must rank in the top N by CURRENT "
                             "absolute capex level to be eligible for the "
                             "growth ranking.")
    parser.add_argument("--min-base-capex", type=float, default=MIN_BASE_CAPEX_USD,
                        help="A base-date capex figure below this (USD) is "
                             "treated as unusable, not a genuine tiny base.")
    parser.add_argument("--min-interval", type=float, default=None,
                        help="Override edgar_data's default self-throttle "
                             "interval (seconds between SEC requests).")
    args = parser.parse_args()

    snapshot_date = args.date or datetime.date.today().isoformat()
    base_date = pd.Timestamp(args.base_asof)
    recent_date = pd.Timestamp(snapshot_date)  # "most recent available" = today
    from mining.edgar_data import DEFAULT_MIN_REQUEST_INTERVAL_SECONDS
    min_interval = args.min_interval or DEFAULT_MIN_REQUEST_INTERVAL_SECONDS

    print(f"Fetching S&P 500 constituents from {SP500_URL} ...")
    sp500_raw = _fetch_sp500_tickers()
    sp500_tickers = sorted({universe.normalize_ticker_for_yfinance(t) for t in sp500_raw})
    print(f"  {len(sp500_raw)} raw rows -> {len(sp500_tickers)} normalized unique tickers")

    print(f"Fetching CIK resolution table from {COMPANY_TICKERS_URL} ...")
    company_tickers_payload = _fetch_company_tickers()
    cik_table = edgar_cik_resolution.parse_company_tickers_payload(company_tickers_payload)
    cik_by_ticker = edgar_cik_resolution.resolve_cik_for_tickers(cik_table, sp500_tickers)
    unresolved = sorted(t for t, cik in cik_by_ticker.items() if cik is None)
    print(f"  resolved {len(sp500_tickers) - len(unresolved)}/{len(sp500_tickers)} "
          f"tickers to a CIK; {len(unresolved)} unresolved: {unresolved}")

    print(f"Screening capex growth ({base_date.date()} -> {recent_date.date()}) via "
          f"EDGAR (self-throttled to {min_interval}s between requests, up to "
          f"{len(CAPEX_TAGS)} fallback tags per company) ...")
    level_by_ticker = {}   # recent (current) absolute level -- the floor input
    growth_by_ticker = {}  # recent / base -- the ranking input
    per_ticker_detail = {}
    resolved_tickers = [t for t in sp500_tickers if cik_by_ticker[t] is not None]
    for i, ticker in enumerate(resolved_tickers):
        cik = cik_by_ticker[ticker]
        if i > 0:
            time.sleep(min_interval)  # continuous rate limit ACROSS companies too
        try:
            series, tags_used = _build_point_in_time_series_for_ticker(
                cik, user_agent=None,
                sleep=time.sleep, min_interval=min_interval)
        except sec_contact.SecContactMissing:
            raise
        except Exception as exc:  # noqa: BLE001 -- one bad filer must not sink the run
            level_by_ticker[ticker] = None
            growth_by_ticker[ticker] = None
            per_ticker_detail[ticker] = {"cik": cik, "status": "error", "reason": str(exc)}
            print(f"  [{i+1}/{len(resolved_tickers)}] {ticker}: ERROR {exc}")
            continue
        # One fetch, two point-in-time evaluations of the SAME series --
        # no second network pass needed to get both the baseline and the
        # current figure.
        base_value = capex_watchlist.trailing_capex_asof(series, base_date)
        recent_value = capex_watchlist.trailing_capex_asof(series, recent_date)
        growth = capex_watchlist.capex_growth_multiple(
            base_value, recent_value, min_base_val=args.min_base_capex)
        level_by_ticker[ticker] = recent_value
        growth_by_ticker[ticker] = growth
        if recent_value is None:
            status = "no_usable_recent_disclosure"
        elif base_value is None:
            status = "no_usable_base_disclosure"  # e.g. IPO'd/no history before base_date
        elif growth is None:
            status = "base_below_min_floor"  # base_value present but too small to trust
        else:
            status = "ok"
        per_ticker_detail[ticker] = {
            "cik": cik, "tags_used": tags_used,
            "base_capex_usd": base_value, "recent_capex_usd": recent_value,
            "capex_growth_multiple": growth, "status": status,
        }
        if (i + 1) % 25 == 0 or i == len(resolved_tickers) - 1:
            shown = f"{growth:.2f}x" if growth is not None else "n/a"
            print(f"  [{i+1}/{len(resolved_tickers)}] {ticker}: growth={shown}")

    # Two tickers can share one CIK (e.g. GOOG/GOOGL,
    # both Alphabet) -- dedupe BEFORE ranking so N reflects N distinct
    # companies, not N tickers. Applied to BOTH dicts via the same
    # surviving ticker set so level/growth stay consistent per company.
    deduped_level_by_ticker = capex_watchlist.dedupe_share_classes_by_cik(
        level_by_ticker, cik_by_ticker)
    dropped_as_duplicate_share_class = sorted(
        set(level_by_ticker) - set(deduped_level_by_ticker))
    deduped_growth_by_ticker = {t: growth_by_ticker[t] for t in deduped_level_by_ticker}
    if dropped_as_duplicate_share_class:
        print(f"  dropped {len(dropped_as_duplicate_share_class)} duplicate share "
              f"class ticker(s) (same CIK as another kept ticker): "
              f"{dropped_as_duplicate_share_class}")

    ranked = capex_watchlist.rank_by_capex_growth(
        deduped_level_by_ticker, deduped_growth_by_ticker,
        level_floor_rank=args.level_floor_rank, n=args.n)
    watchlist_tickers = [row["ticker"] for row in ranked]

    result = {
        "tickers": watchlist_tickers,
        "n_requested": args.n,
        "n_selected": len(watchlist_tickers),
        "base_date": str(base_date.date()),
        "recent_date": str(recent_date.date()),
        "level_floor_rank": args.level_floor_rank,
        "min_base_capex_usd": args.min_base_capex,
        "dropped_as_duplicate_share_class": dropped_as_duplicate_share_class,
        "known_limitations": [
            "Growth ranking requires a usable capex disclosure at BOTH base_date "
            "and recent_date -- a company with no history before base_date (a "
            "post-2019 IPO or spinoff) has an undefined growth multiple and is "
            "excluded from the growth ranking entirely, however large its recent "
            "capex ramp, since there is no baseline to measure growth FROM. This "
            "is a real, disclosed gap in coverage, not a bug: a genuinely new "
            "company with an explosive AI-infrastructure capex program could be "
            "missed by this specific methodology.",
            "min_base_capex_usd excludes a base-date figure below the floor as "
            "unreliable rather than genuinely tiny. A filer can disclose a "
            "very small value for a period and later restate the identical "
            "period to a different value (a filer inconsistency: the SAME "
            "period, two values). A growth RATIO is far more sensitive to a "
            "spuriously small denominator than a level ranking is, so such a "
            "base can manufacture a very large 'growth' figure with no real "
            "acceleration behind it. The floor is a blunt instrument -- it also "
            "excludes any company with a genuinely small-but-real base near the "
            "threshold. The ranked list is not checked name-by-name against "
            "raw EDGAR data.",
            "Tag-conflict resolution (mining.capex_watchlist.resolve_tag_conflicts): "
            "when the same period and filing is disclosed under more than one "
            "capex tag with different values, the tag with a restatement in "
            "its own recent history (within 3 years of the period being "
            "resolved, using only what was knowable at each row's own filed "
            "date -- no lookahead) loses to a tag with none, regardless of "
            "magnitude. When neither or both sides show such instability, the "
            "LARGER value wins, and CAPEX_TAGS's listed order breaks an exact "
            "value tie. Rationale: the narrowest, most-standard tag can capture "
            "only a small subset of a company's real capital spending, so "
            "undercounting is the worse failure mode here; but a larger tag "
            "that is repeatedly restated may be capturing lumpy acquisitions "
            "rather than organic capex, which the stability check demotes. "
            "This rule is not proven exhaustive: other tag-mismatch patterns "
            "may exist elsewhere in the universe, and some companies have no "
            "data under most CAPEX_TAGS, or far understate their real spend, "
            "so they cannot be ranked correctly.",
        ],
        "selection_rule": "PROVISIONAL_DEFAULT -- among S&P 500 "
                          "constituents in the top level_floor_rank by CURRENT "
                          "(recent_date) absolute trailing annual capex (one per "
                          "CIK -- a company with multiple share-class tickers, "
                          "e.g. GOOG/GOOGL, is deduped to its alphabetically-"
                          "first ticker before ranking), the N with the largest "
                          "capex GROWTH multiple (recent_date figure / base_date "
                          "figure). SEC XBRL, fallback tag union, filed-date "
                          "point-in-time, never end-date, for both figures. "
                          "N, base_date, and level_floor_rank are disclosed "
                          "defaults, not yet validated; recent_date is always "
                          "the run date. Ranked by growth rather than pure "
                          "level (the thesis is capex that's "
                          "ACCELERATING due to AI/infrastructure spend, not "
                          "just large in absolute terms).",
        "universe_source": "live S&P 500 Wikipedia table fetch, current constituents "
                           "only (no point-in-time index-membership history exists "
                           "in this repo) -- survivorship-bias caveat, same "
                           "convention as every other pooled family here",
        "sp500_source_url": SP500_URL,
        "cik_source_url": COMPANY_TICKERS_URL,
        "sec_contact_source": "env:" + sec_contact.ENV_VAR,
        "capex_tags_tried": list(CAPEX_TAGS),
        "fetch_date": snapshot_date,
        "sp500_universe_count": len(sp500_tickers),
        "cik_unresolved_count": len(unresolved),
        "cik_unresolved_tickers": unresolved,
        "ranked": ranked,
        "per_ticker_detail": per_ticker_detail,
    }

    out_path = os.path.join(args.out_dir, f"capex_watchlist_{snapshot_date}.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=False)

    print(f"\nSelected {len(watchlist_tickers)}/{args.n} names: {watchlist_tickers}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
