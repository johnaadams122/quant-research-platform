"""Re-derivable fetcher for the batch5 candidate
``ex_dividend_day_capture_pooled``'s event table.

Writes ``data/mining/batch5_events/ex_dividend_day_capture_pooled/``
``events.json`` + ``provenance.json`` from one yfinance call per pool member:

    yf.Ticker(t).history(period="max", auto_adjust=False, actions=True)

That single response supplies all three inputs, so the dividend amount, the
signal-bar close and the corporate-action flags always come from ONE
point-in-time view of Yahoo's history rather than three independent pulls
that could disagree.

Data hazards this script handles
--------------------------------
An event table built from this source without these checks carries three
hazards:

1. **A split-adjustment race.** Yahoo adjusts its *dividend* series and its
   *price* history for a split independently, and the two sides can
   propagate at different times with no established ordering (for the
   Amphenol 2:1 split of 2026-09-03 the dividend series was adjusted while
   the price history was not). Dividends adjusted first, prices lagging,
   understates every pre-split yield by the split ratio (conservative). The
   reverse order -- prices adjusted first, amounts still on the pre-split
   basis -- leaves no step across the split session at all, while it
   OVERSTATES every pre-split yield by the split ratio and pushes events
   above the floor. yfinance does not repair either series on this code path
   (its split-repair routines run only under a repair flag this fetcher does
   not pass), so both orders are possible. ``verify_split_basis`` below
   checks BOTH directions -- the price history's basis and the dividend
   series' own basis -- reports, per direction, what it decided and what it
   could not, and the fetcher REFUSES to write a file that shows a violation
   (fail closed).
2. **Non-cash distributions carried as dividends.** Yahoo's Dividends column
   mixes spin-off, recapitalization and merger distributions in with cash
   dividends. For a spin-off the stored "amount" is the market value of the
   spun-off shares, fixed on or after the ex-date and therefore unknowable at
   the E-2 signal bar, and the auto-adjusted ex-day return on such a day is
   an adjustment artifact rather than dividend capture. This script records
   the MECHANISM evidence per event (Yahoo's own same-session split-type
   adjustment factor, plus the issuer's trailing dividend series) and the
   module applies the exclusion; nothing is dropped silently here.
3. **Staleness.** Yahoo can post an ex-date after the session it falls on,
   so a fetch made soon after the derived holdout window ends can miss real
   ex-dates on its last session; re-fetch once those sessions are posted.

Evidence fields written per event (every one computable from data <= E, so a
consumer can filter causally):

    split_on_ex_date        Yahoo's "Stock Splits" value on session E (0.0 =
                            none). A spin-off recorded as a dividend is
                            frequently accompanied by a same-session
                            split-type adjustment factor.
    prior_amount            the issuer's immediately preceding dividend
                            amount (null for its first ever dividend)
    trailing_median_amount  median of up to the 4 preceding dividend amounts
    prior_count             how many of the issuer's dividends precede this
                            one

Usage:
    python \\
        scripts/fetch_batch5_events_ex_dividend_day_capture_pooled.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import statistics
import sys
import time
from fractions import Fraction

import pandas as pd
import yfinance as yf

FAMILY_ID = "ex_dividend_day_capture_pooled"
SCHEMA_VERSION = 3
SIGNAL_SESSIONS_BEFORE_EX = 2
TRAILING_WINDOW = 4
# Yahoo throttles a 500-member serial pull: an unretried run can lose members
# to transient empty responses, which would silently shrink the pooled sample.
MAX_ATTEMPTS = 4
RETRY_SLEEP_SECONDS = 6.0

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULT_SNAPSHOT = os.path.join(
    REPO_ROOT, "data", "mining", "universe_snapshot_2026-08-29.json")
DEFAULT_OUT_DIR = os.path.join(
    REPO_ROOT, "data", "mining", "batch5_events", FAMILY_ID)

# --- split-basis cross-check (BOTH directions) ------------------------------
# Yahoo adjusts its dividend series and its price history for a split
# INDEPENDENTLY, and the two sides can propagate at different times with no
# established ordering (see the Amphenol split above). Either order
# mis-scales the yield, in OPPOSITE directions, so both are read here:
#
#   A. price history not yet adjusted, dividend series already adjusted.
#      close[S] / close[S-1] lands on the raw 1/ratio step, because the
#      pre-split closes were never divided. Every pre-split yield is then
#      UNDERSTATED by the ratio (conservative: events fall below the floor).
#      This is the APH direction, and the price step is its footprint.
#   B. price history already adjusted, dividend series not yet adjusted.
#      The close series steps normally across S, so direction A's test finds
#      NOTHING and passes silently -- yet every pre-split yield is
#      OVERSTATED by the ratio (ANTI-conservative: it pushes events above the
#      floor that do not belong there).
#
# B is invisible in the price series by construction, so it needs an
# independent handle on the dividend side's own basis. The issuer's own
# dividend series is one: pre-split amounts still on the raw basis sit a full
# ``ratio`` above their post-split neighbours, so the median of the dividends
# just before S divided by the median of those just after S lands ON the
# ratio. What the reading proves is therefore one-sided but exact about its
# own limits: a step inside the raw-basis band is the direction-B footprint,
# and a step outside it REFUTES direction B for that split (a raw basis
# would have to be cancelled by a dividend change of the split ratio itself).
# It is usable only when (i) the factor is a genuine ANNOUNCED share ratio,
# for the arithmetic reason spelled out at ``ANNOUNCED_MAX_DENOMINATOR``
# below, (ii) the raw-basis band is separated from an ordinary flat series by
# ``DIV_SEPARATION_MARGIN``, so a small dividend cut cannot be read as a raw
# basis, and (iii) the issuer paid dividends on both sides of S, near it in
# time. Every split failing any of those is counted as UNDECIDED and
# reported -- never assumed clean -- and a split with no dividend before it
# at all is recorded separately, since there is then no pre-split amount for
# direction B to mis-scale.
BASIS_RATIO_TOLERANCE = 0.015   # |observed / (1/ratio) - 1| <= this => unapplied
BASIS_MATERIAL_STEP = 0.10      # only ratios whose 1/ratio differs from 1 by
                                # more than this are decidable (a 1.02 stock
                                # dividend is indistinguishable from a -2% day)
PRICE_APPLIED_BAND = 0.10       # |close step - 1| <= this => positively read
                                # as applied; anything else is inconclusive
                                # rather than silently counted as clean
DIV_NEIGHBOUR_COUNT = 4         # dividends compared on each side of a split
DIV_NEIGHBOUR_MAX_SESSIONS = 400  # ...and only within this many sessions, so
                                # the comparison is local to the split
DIV_RATIO_TOLERANCE = 0.15      # |step / ratio - 1| <= this => pre-split
                                # (raw) amounts; wider than the price
                                # tolerance because a dividend raise often
                                # lands on the split itself
DIV_SEPARATION_MARGIN = 0.20    # the raw-basis band must clear 1.0 by this
                                # much, else an ordinary dividend change
                                # could be misread as a raw basis (undecided)
# Direction B is only READABLE on a genuine share split, and the reason is
# arithmetic, not stylistic. Write C for the issuer's true per-share dividend
# change across the session (raw pre-amount / raw post-amount). With the
# dividend series correctly rebased the observed step is C/ratio; with the
# pre-split amounts still raw it is C. For a real split the company rescales
# its per-share dividend BY the split, so C ~ ratio: the no-race step sits at
# 1 and the race step at ``ratio``, well separated. For a value-derived
# distribution factor (a spin-off's price-ratio adjustment) nothing ties C to
# the factor, so neither the null nor the alternative is pinned and the band
# is aimed at the wrong place. Worked example: PNR 2018-05-01 (the nVent
# spin-off) carries factor 1.489; Yahoo rebased the pre-spin amounts
# (2018-Q1 shows 0.235057 = 0.35 / 1.489) and its prices step 0.9919 across
# the session, so BOTH series are on one basis -- yet Pentair's genuine
# post-spin dividend cut (raw 0.35 -> 0.175, C = 2.0) puts the observed step
# at C/1.489 = 1.34, inside a band centred on 1.489. A band test there would
# report a race that is not present, while a true race would show
# step = C = 2.0, OUTSIDE the band: the test would point away from its own
# alternative. So a value-derived factor is not read at all.
ANNOUNCED_MAX_DENOMINATOR = 100  # 2:1, 3:1, 3:2, 1:2, 1:4, 5%/10% stock divs
ANNOUNCED_TOLERANCE = 1e-6       # the factor is reported to ~6 decimals, so
                                 # an exact p/q ratio can only miss by
                                 # print rounding at this scale


def _log(message: str) -> None:
    print(message, flush=True)


def load_universe(path: str) -> list:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    tickers = payload.get("tickers")
    if not isinstance(tickers, list) or not tickers:
        raise ValueError(f"{path}: 'tickers' must be a non-empty list")
    return list(tickers)


def fetch_history(ticker: str):
    """One yfinance response per member: split-adjusted, dividend-UNadjusted
    OHLC plus the Dividends and Stock Splits action columns."""
    frame = yf.Ticker(ticker).history(
        period="max", auto_adjust=False, actions=True)
    if frame is None or frame.empty:
        return None
    frame = frame.copy()
    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    frame.index = index.normalize()
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    for column in ("Dividends", "Stock Splits"):
        if column not in frame:
            frame[column] = 0.0
    return frame


def _empty_basis_summary() -> dict:
    """Zeroed per-member tallies for ``verify_split_basis``."""
    return {
        "splits_seen": 0,
        "splits_material": 0,
        "splits_immaterial_ratio": 0,
        # direction A: the price history's own basis
        "price_applied": 0,
        "price_unapplied": 0,
        "price_inconclusive": 0,
        "price_undecidable": 0,
        # direction B: the dividend series' own basis
        "dividend_refuted_raw_basis": 0,
        "dividend_unadjusted": 0,
        "dividend_no_prior_dividend": 0,
        # a value-derived distribution factor: direction B is not readable
        # there at all (see ANNOUNCED_MAX_DENOMINATOR's comment)
        "dividend_not_read_value_derived_factor": 0,
        "dividend_undecidable_ratio_not_separable": 0,
        "dividend_undecidable_too_few_before": 0,
        # the live-race window: a split too recent to have two dividends after
        # it yet is exactly when a propagation race is in flight, and direction
        # B is structurally blind there (see _dividend_basis_reading)
        "dividend_undecidable_too_few_after": 0,
        # how many dividend records sit BEFORE a split whose dividend-side
        # basis could not be decided (the events direction B could still be
        # mis-scaling undetected)
        "events_before_an_undecided_split": 0,
        # ...and WHICH sessions those were, so the disclosure is auditable
        # instead of a bare count a reader has to take on trust
        "undecided_splits": [],
    }


def _price_basis_reading(closes, position: int, ratio: float) -> tuple:
    """``(reading, observed_step)`` for direction A at frame ``position``.

    ``"unapplied"`` is the positive detection: the close step landed on the
    raw ``1/ratio`` the split would have removed. ``"applied"`` is the
    positive read the other way (an ordinary daily return). A step that is
    neither is ``"inconclusive"`` -- a split day can legitimately move a lot
    -- and is reported as such rather than counted as clean.
    """
    previous = float(closes.iloc[position - 1])
    current = float(closes.iloc[position])
    if not (math.isfinite(previous) and math.isfinite(current)) or previous <= 0.0:
        return "undecidable", None
    observed = current / previous
    unapplied_step = 1.0 / ratio
    if abs(observed / unapplied_step - 1.0) <= BASIS_RATIO_TOLERANCE:
        return "unapplied", observed
    if abs(observed - 1.0) <= PRICE_APPLIED_BAND:
        return "applied", observed
    return "inconclusive", observed


def _raw_basis_band(ratio: float) -> tuple:
    return ratio * (1.0 - DIV_RATIO_TOLERANCE), ratio * (1.0 + DIV_RATIO_TOLERANCE)


def is_announced_share_ratio(factor: float) -> bool:
    """True for a factor that is a simple ANNOUNCED share ratio.

    2-for-1, 3-for-1, 3-for-2, 1-for-2, 1-for-4 and the percentage stock
    dividends all reduce to p/q with a small q; a value-derived vendor
    adjustment factor (a spin-off's pre-over-post distribution price ratio,
    e.g. DHR 2016-07-05's 1.319 or PNR 2018-05-01's 1.489) does not.
    """
    if not math.isfinite(factor) or factor <= 0.0:
        return False
    fraction = Fraction(factor).limit_denominator(ANNOUNCED_MAX_DENOMINATOR)
    return abs(float(fraction) - factor) <= ANNOUNCED_TOLERANCE * max(1.0, factor)


def _dividend_basis_reading(paying: list, amounts: list, position: int,
                            ratio: float) -> tuple:
    """``(reading, observed_step, detail)`` for direction B at ``position``.

    ``"unadjusted"`` is the positive detection of direction B: the issuer's
    dividends immediately before the split sit a factor ``ratio`` above those
    immediately after, which is what a raw pre-split basis looks like.
    ``"refuted_raw_basis"`` means the step is outside that band, so the
    pre-split amounts cannot be on the raw basis unless a dividend change of
    the split ratio itself coincided with the split. The remaining readings
    are the honest undecided cases.

    ``"undecidable_too_few_after"`` is the important one: a split too recent
    for the issuer to have paid two dividends since is exactly when a
    propagation race is in flight, and this reading is BLIND there. Direction
    B rescales every pre-split dividend uniformly, so with no post-split
    dividend to compare against, the member's own data carries no contrast
    that could expose it -- the yield series is inflated end to end. There is
    no fix for that inside this data source; the count is reported instead.
    """
    prior = [pos for pos in paying if pos < position]
    detail = {"pre_split_dividends": len(prior)}
    if not prior:
        return "no_prior_dividend", None, detail
    if not is_announced_share_ratio(ratio):
        return "not_read_value_derived_factor", None, detail
    low, high = _raw_basis_band(ratio)
    if not (high < 1.0 - DIV_SEPARATION_MARGIN or low > 1.0 + DIV_SEPARATION_MARGIN):
        return "undecidable_ratio_not_separable", None, detail
    before = [amount for pos, amount in zip(paying, amounts)
              if pos < position and position - pos <= DIV_NEIGHBOUR_MAX_SESSIONS]
    after = [amount for pos, amount in zip(paying, amounts)
             if pos > position and pos - position <= DIV_NEIGHBOUR_MAX_SESSIONS]
    detail["neighbours_before"] = len(before)
    detail["neighbours_after"] = len(after)
    if len(after) < 2:
        return "undecidable_too_few_after", None, detail
    if len(before) < 2:
        return "undecidable_too_few_before", None, detail
    median_before = statistics.median(before[-DIV_NEIGHBOUR_COUNT:])
    median_after = statistics.median(after[:DIV_NEIGHBOUR_COUNT])
    if not (math.isfinite(median_before) and math.isfinite(median_after)) \
            or median_after <= 0.0:
        return "undecidable_too_few_after", None, detail
    step = median_before / median_after
    if low <= step <= high:
        return "unadjusted", step, detail
    return "refuted_raw_basis", step, detail


def verify_split_basis(ticker: str, frame) -> tuple:
    """Return ``(summary, violations)`` -- the two-directional basis check.

    A violation is a split session where Yahoo's dividend series and its own
    price history are demonstrably on DIFFERENT split bases, in either
    direction (see the module comment above ``BASIS_RATIO_TOLERANCE``):

    * ``price_history_unapplied`` -- the close step still carries the raw
      1/ratio the split would have removed (the APH direction; understates
      pre-split yields).
    * ``dividend_series_unadjusted`` -- the dividends just before the split
      sit a factor ``ratio`` above their post-split neighbours while the
      price history steps normally (the reverse direction; OVERSTATES
      pre-split yields and pushes events above the floor).

    An empty violation list does NOT mean "no race is possible": it means
    neither footprint was found among the splits this test could decide, and
    ``summary`` says exactly how many it could not.
    """
    summary = _empty_basis_summary()
    violations = []
    splits = frame["Stock Splits"]
    closes = frame["Close"]
    dividends = frame["Dividends"]
    paying = [position for position in range(len(frame))
              if float(dividends.iloc[position]) > 0.0]
    amounts = [float(dividends.iloc[position]) for position in paying]
    undecided_event_positions = set()
    for position in range(1, len(frame)):
        ratio = float(splits.iloc[position])
        if ratio <= 0.0 or ratio == 1.0 or not math.isfinite(ratio):
            continue
        summary["splits_seen"] += 1
        if abs(1.0 / ratio - 1.0) <= BASIS_MATERIAL_STEP:
            summary["splits_immaterial_ratio"] += 1
            continue
        summary["splits_material"] += 1
        session = frame.index[position].strftime("%Y-%m-%d")
        price_reading, observed = _price_basis_reading(closes, position, ratio)
        summary[{"applied": "price_applied", "unapplied": "price_unapplied",
                 "inconclusive": "price_inconclusive",
                 "undecidable": "price_undecidable"}[price_reading]] += 1
        dividend_reading, step, detail = _dividend_basis_reading(
            paying, amounts, position, ratio)
        summary[{
            "refuted_raw_basis": "dividend_refuted_raw_basis",
            "unadjusted": "dividend_unadjusted",
            "no_prior_dividend": "dividend_no_prior_dividend",
            "not_read_value_derived_factor":
                "dividend_not_read_value_derived_factor",
            "undecidable_ratio_not_separable":
                "dividend_undecidable_ratio_not_separable",
            "undecidable_too_few_before":
                "dividend_undecidable_too_few_before",
            "undecidable_too_few_after":
                "dividend_undecidable_too_few_after",
        }[dividend_reading]] += 1
        if dividend_reading not in ("refuted_raw_basis", "unadjusted",
                                    "no_prior_dividend"):
            # direction B reached no verdict here: every dividend record
            # before this split session is exposure it did not clear
            undecided_event_positions.update(
                pos for pos in paying if pos < position)
            summary["undecided_splits"].append({
                "ticker": ticker,
                "split_session": session,
                "split_ratio": ratio,
                "reading": dividend_reading,
                "events_before": int(detail.get("pre_split_dividends", 0)),
            })
        if price_reading == "unapplied":
            violations.append({
                "ticker": ticker,
                "kind": "price_history_unapplied",
                "split_session": session,
                "split_ratio": ratio,
                "observed_close_step": observed,
                "unapplied_step_would_be": 1.0 / ratio,
                "dividend_side_reading": dividend_reading,
                "dividend_side_step": step,
                "reading": ("the price history still carries the raw pre-split "
                            "step, so its closes and the dividend amounts are "
                            "on different split bases unless the dividend "
                            "series is equally unadjusted; pre-split yields "
                            "for this member are mis-scaled by the ratio"),
            })
        elif dividend_reading == "unadjusted":
            violations.append({
                "ticker": ticker,
                "kind": "dividend_series_unadjusted",
                "split_session": session,
                "split_ratio": ratio,
                "observed_close_step": observed,
                "price_side_reading": price_reading,
                "dividend_side_step": step,
                "dividend_neighbours": detail,
                "reading": ("the dividends immediately before this split sit a "
                            "factor ratio above their post-split neighbours "
                            "while the price history steps normally across the "
                            "session: the amounts are still on the PRE-split "
                            "basis and the closes are not, so every pre-split "
                            "yield for this member is OVERSTATED by the ratio "
                            "-- the anti-conservative direction, which pushes "
                            "events above the yield floor"),
            })
    summary["events_before_an_undecided_split"] = len(undecided_event_positions)
    return summary, violations


def build_records(frame) -> list:
    """Event records for one member, in ex-date order."""
    dividends = frame["Dividends"]
    paying = [position for position in range(len(frame))
              if float(dividends.iloc[position]) > 0.0]
    closes = frame["Close"]
    splits = frame["Stock Splits"]
    amounts = [float(dividends.iloc[position]) for position in paying]
    records = []
    for order, position in enumerate(paying):
        amount = amounts[order]
        signal_position = position - SIGNAL_SESSIONS_BEFORE_EX
        if signal_position >= 0:
            signal_date = frame.index[signal_position].strftime("%Y-%m-%d")
            signal_close = float(closes.iloc[signal_position])
        else:
            signal_date, signal_close = None, None
        trailing = amounts[max(0, order - TRAILING_WINDOW):order]
        records.append({
            "ex_date": frame.index[position].strftime("%Y-%m-%d"),
            "amount": amount,
            "signal_date": signal_date,
            "unadjusted_close_signal": signal_close,
            "yield": ((amount / signal_close)
                      if signal_close and signal_close > 0.0 else None),
            "split_on_ex_date": float(splits.iloc[position]),
            "prior_amount": (amounts[order - 1] if order > 0 else None),
            "trailing_median_amount": (statistics.median(trailing)
                                       if trailing else None),
            "prior_count": order,
        })
    return records


def fetch_history_with_retry(ticker: str, *, attempts: int = MAX_ATTEMPTS,
                             sleep_seconds: float = RETRY_SLEEP_SECONDS) -> tuple:
    """``(frame, reason)`` -- retries a transient empty/failed response with a
    linear backoff. ``frame is None`` means every attempt failed, and
    ``reason`` then names the last failure."""
    reason = None
    for attempt in range(1, attempts + 1):
        try:
            frame = fetch_history(ticker)
        except Exception as exc:                     # noqa: BLE001 - reported
            frame, reason = None, f"{type(exc).__name__}: {exc}"
        else:
            if frame is not None:
                return frame, None
            reason = "yfinance returned no history"
        if attempt < attempts:
            time.sleep(sleep_seconds * attempt)
    return None, reason


def fetch_all(tickers: list) -> tuple:
    events, coverage, failures, violations = {}, {}, {}, []
    basis = _empty_basis_summary()
    members_with_an_undecided_split = []
    for number, ticker in enumerate(tickers, start=1):
        frame, reason = fetch_history_with_retry(ticker)
        if frame is None:
            failures[ticker] = reason
            _log(f"  [{number}/{len(tickers)}] {ticker}: FAILED {reason}")
            continue
        member_basis, member_violations = verify_split_basis(ticker, frame)
        for key, value in member_basis.items():
            if isinstance(value, list):
                basis[key].extend(value)
            else:
                basis[key] += value
        if member_basis["events_before_an_undecided_split"]:
            members_with_an_undecided_split.append(ticker)
        violations.extend(member_violations)
        records = build_records(frame)
        events[ticker] = records
        coverage[ticker] = {
            "first_session": frame.index[0].strftime("%Y-%m-%d"),
            "last_session": frame.index[-1].strftime("%Y-%m-%d"),
            "sessions": int(len(frame)),
        }
        if number % 25 == 0 or number == len(tickers):
            _log(f"  [{number}/{len(tickers)}] {ticker}: {len(records)} events")
    basis["members_with_an_undecided_split"] = sorted(
        members_with_an_undecided_split)
    return events, coverage, failures, violations, basis


DIRECTION_A = "price_history_unapplied"
DIRECTION_B = "dividend_series_unadjusted"


def basis_check_record(basis: dict) -> dict:
    """The ``basis_check`` block written into events.json and provenance.json.

    Deliberately states its OWN limits: ``proves`` is what the two readings
    established, ``does_not_prove`` is the residue the module's docstring
    must carry as a disclosed limitation rather than a guarantee.
    ``load_events`` refuses a table whose ``directions`` does not
    carry BOTH keys, so a one-directional check cannot pass as this one.
    """
    return {
        "status": "verified",
        "directions": {
            DIRECTION_A: {
                "detects": ("a split Yahoo applied to its dividend series but "
                            "NOT to its price history: the close step across "
                            "the split session still carries the raw 1/ratio. "
                            "Understates pre-split yields (conservative)."),
                "splits_read_as_applied": basis["price_applied"],
                "splits_read_as_unapplied": basis["price_unapplied"],
                "splits_inconclusive": basis["price_inconclusive"],
                "splits_undecidable": basis["price_undecidable"],
            },
            DIRECTION_B: {
                "detects": ("the REVERSE race -- prices already split-adjusted "
                            "while the dividend amounts are still on the "
                            "pre-split basis. Leaves no step in the price "
                            "series, so direction A cannot see it. OVERSTATES "
                            "pre-split yields by the ratio (anti-conservative: "
                            "pushes events above the yield floor). Read from "
                            "the issuer's own dividend series: pre-split "
                            "amounts on a raw basis sit a factor ratio above "
                            "their post-split neighbours."),
                "splits_refuting_a_raw_dividend_basis":
                    basis["dividend_refuted_raw_basis"],
                "splits_read_as_unadjusted": basis["dividend_unadjusted"],
                "splits_with_no_prior_dividend":
                    basis["dividend_no_prior_dividend"],
                "splits_not_read_value_derived_factor":
                    basis["dividend_not_read_value_derived_factor"],
                "splits_undecidable_ratio_not_separable":
                    basis["dividend_undecidable_ratio_not_separable"],
                "splits_undecidable_too_few_dividends_before":
                    basis["dividend_undecidable_too_few_before"],
                "splits_undecidable_too_few_dividends_after":
                    basis["dividend_undecidable_too_few_after"],
                "blind_spot": (
                    "splits_undecidable_too_few_dividends_after is the live "
                    "race window and this reading is structurally blind there: "
                    "direction B rescales EVERY pre-split dividend uniformly, "
                    "so without a post-split dividend to compare against, the "
                    "member's own data carries no contrast that could expose "
                    "it. Not fixable inside this data source"),
                "events_before_an_undecided_split":
                    basis["events_before_an_undecided_split"],
                "members_with_an_undecided_split":
                    basis.get("members_with_an_undecided_split", []),
                "undecided_splits": sorted(
                    basis["undecided_splits"],
                    key=lambda row: (row["ticker"], row["split_session"])),
            },
        },
        "proves": ("for every split session this check could decide, neither "
                   "the price-side nor the dividend-side footprint of a "
                   "split-basis race is present, so the dividend amounts and "
                   "the signal-bar closes share one split basis there"),
        "does_not_prove": (
            "(1) splits whose ratio is within material_step of 1 are not read "
            "at all in either direction; (2) direction A reads 'applied' "
            "positively only when the close step is within "
            "price_applied_band of 1 -- an inconclusive step is reported, not "
            "counted clean; (3) direction B is undecided for a split whose "
            "raw-basis band is not separated from 1 by "
            "dividend_separation_margin, or with fewer than two dividends "
            "within dividend_neighbour_max_sessions on either side, or whose "
            "factor is value-derived rather than an announced share ratio "
            "(nothing then ties the issuer's own dividend change to the "
            "factor, so the reading is aimed at the wrong place -- see "
            "splits_not_read_value_derived_factor), and "
            "events_before_an_undecided_split counts the dividend records "
            "that sit behind any such split; (4) in particular direction B is "
            "STRUCTURALLY BLIND to a split too recent to have two dividends "
            "after it -- the live race window -- because it rescales every "
            "pre-split amount uniformly and leaves no internal contrast; "
            "(5) neither reading can separate a genuine dividend change of "
            "exactly the split ratio from a raw basis"),
        "splits_seen": basis["splits_seen"],
        "splits_checked": basis["splits_material"],
        "splits_immaterial_ratio": basis["splits_immaterial_ratio"],
        "ratio_tolerance": BASIS_RATIO_TOLERANCE,
        "material_step": BASIS_MATERIAL_STEP,
        "price_applied_band": PRICE_APPLIED_BAND,
        "dividend_ratio_tolerance": DIV_RATIO_TOLERANCE,
        "dividend_separation_margin": DIV_SEPARATION_MARGIN,
        "dividend_neighbour_count": DIV_NEIGHBOUR_COUNT,
        "dividend_neighbour_max_sessions": DIV_NEIGHBOUR_MAX_SESSIONS,
        "violations": [],
    }


def summarize(events: dict) -> dict:
    total = sum(len(records) for records in events.values())
    with_yield = sum(1 for records in events.values() for record in records
                     if record["yield"] is not None)
    split_flagged = sum(1 for records in events.values() for record in records
                        if record["split_on_ex_date"] not in (0.0, 1.0))
    return {
        "event_count": total,
        "events_with_yield": with_yield,
        "events_with_same_session_split": split_flagged,
        "members_with_events": sum(1 for records in events.values() if records),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--reason", default=(
        "schema-3 fetch: verify_split_basis reads both split-basis "
        "directions -- the price history's own basis (dividends adjusted, "
        "prices lagging, which understates a yield) and the dividend "
        "series' own basis (prices adjusted, amounts still on the pre-split "
        "basis, which OVERSTATES every pre-split yield by the split ratio) "
        "-- and the basis_check block records, per direction, what was "
        "decided and what could not be"))
    parser.add_argument("--limit", type=int, default=0,
                        help="fetch only the first N members (smoke runs)")
    parser.add_argument("--allow-failures", action="store_true",
                        help=("write the table even though some members could "
                              "not be fetched. OFF by default: a member absent "
                              "from events.json contributes no trades, so a "
                              "partial fetch silently shrinks the pooled "
                              "sample instead of failing."))
    args = parser.parse_args(argv)

    started = dt.datetime.now(dt.timezone.utc).isoformat()
    tickers = load_universe(args.snapshot)
    if args.limit:
        tickers = tickers[:args.limit]
    _log(f"{FAMILY_ID}: fetching {len(tickers)} members")
    events, coverage, failures, violations, basis = fetch_all(tickers)

    if violations:
        for violation in violations:
            _log(f"  BASIS VIOLATION [{violation['kind']}] "
                 f"{violation['ticker']} {violation['split_session']} "
                 f"ratio={violation['split_ratio']} "
                 f"close step={violation.get('observed_close_step')} "
                 f"dividend step={violation.get('dividend_side_step')}")
        raise SystemExit(
            f"{FAMILY_ID}: REFUSING to write the event table -- "
            f"{len(violations)} split-basis violation(s): Yahoo's dividend "
            "series and its price history are on different split bases for the "
            "member(s) named above, so their yields would be silently "
            "mis-scaled (understated by the ratio for a "
            "price_history_unapplied reading, OVERSTATED for a "
            "dividend_series_unadjusted one). Re-fetch once Yahoo has "
            "propagated the split to both series.")

    if failures and not args.allow_failures:
        raise SystemExit(
            f"{FAMILY_ID}: REFUSING to write the event table -- "
            f"{len(failures)} member(s) could not be fetched after "
            f"{MAX_ATTEMPTS} attempts each: {sorted(failures)}. A member "
            "absent from events.json produces no trades, so writing now would "
            "silently shrink the pooled sample. Re-run, or pass "
            "--allow-failures deliberately.")

    counts = summarize(events)
    os.makedirs(args.out_dir, exist_ok=True)
    payload = {
        "family_id": FAMILY_ID,
        "schema_version": SCHEMA_VERSION,
        "record_fields": ["ex_date", "amount", "signal_date",
                          "unadjusted_close_signal", "yield",
                          "split_on_ex_date", "prior_amount",
                          "trailing_median_amount", "prior_count"],
        "basis_check": basis_check_record(basis),
        "events": events,
    }
    events_path = os.path.join(args.out_dir, "events.json")
    with open(events_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)

    provenance = {
        "family_id": FAMILY_ID,
        "schema_version": SCHEMA_VERSION,
        "fetch_started_at": started,
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "regenerated_by": ("scripts/fetch_batch5_events_"
                           f"{FAMILY_ID}.py"),
        "reason": args.reason,
        "source": ("yfinance {version} Ticker(t).history(period='max', "
                   "auto_adjust=False, actions=True): the Dividends column "
                   "(ex-dividend dates, split-adjusted amounts), the "
                   "split-adjusted Close on the session two before each "
                   "ex-date, and the Stock Splits column -- all from ONE "
                   "response per member").format(version=yf.__version__),
        "source_url": "https://query2.finance.yahoo.com/ (via yfinance)",
        "universe_file": os.path.relpath(args.snapshot, REPO_ROOT).replace(os.sep, "/"),
        "member_count": len(tickers),
        "members_failed": sorted(failures),
        "failures": failures,
        "date_semantics": (
            "ex_date = yfinance Dividends index (the ex-dividend date); "
            "signal_date = the session two before ex_date on the ticker's own "
            "yfinance calendar; yield = amount / Close on signal_date"),
        "point_in_time": (
            "ex-dates are public at least ten calendar days before the record "
            "date (NYSE 204.12 / Nasdaq 5250(e)(6)); the yield uses only the "
            "signal bar's own close. Every evidence field "
            "(split_on_ex_date, prior_amount, trailing_median_amount, "
            "prior_count) is computable from data <= ex_date."),
        "basis_check": basis_check_record(basis),
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "pandas": pd.__version__,
        "yfinance": yf.__version__,
        "per_ticker_coverage": coverage,
        **counts,
    }
    with open(os.path.join(args.out_dir, "provenance.json"), "w",
              encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=1, sort_keys=True)

    _log(f"wrote {events_path}: {counts}")
    if failures:
        _log(f"members failed: {sorted(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
