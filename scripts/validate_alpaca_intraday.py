"""validate_alpaca_intraday.py
------------------------------
REAL Alpaca API calls -- NOT part of pytest, real credentials from the real
environment (ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY), real network.

The intraday client's defaults rest on measured vendor behavior rather than
assumptions; this script is that measurement.

Measures two things using the real src/mining/intraday_data.py client:

1. Alpaca's actual SIP historical retention depth for QQQ and TQQQ (the
   tickers this pipeline's daily-bar catalog already trades most) --
   bisects a multi-decade window down to a ~1-week bracket around the
   earliest date real minute-bar data comes back for each ticker.

2. Split-adjustment behavior -- fetches NVDA minute bars spanning its real,
   verified 10-for-1 split (effective after close 2024-06-07; split-
   adjusted trading began 2024-06-10 -- confirmed via NVIDIA's own 2024
   Stock Split Investor FAQ and its SEC Form 8-K filed 2024-05-22) with
   adjustment="raw" and separately adjustment="split", and compares the
   Friday-close-to-Monday-open ratio in each to determine which value
   fetch_intraday_bars should request by default.

Writes dated, exact findings to runs/alpaca_intraday_validation.md (the
run date is in the report heading).

Usage:
    python scripts\\validate_alpaca_intraday.py
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import requests  # noqa: E402

from mining.intraday_data import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    IntradayDataGapError,
    fetch_intraday_bars,
)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_FINDINGS_PATH = os.path.join(_REPO_ROOT, "runs", "alpaca_intraday_validation.md")

_RETENTION_SYMBOLS = ["QQQ", "TQQQ"]
_ADJUSTMENT_SYMBOL = "NVDA"
# Real, verified split: announced 2024-05-22, effective after close
# 2024-06-07, split-adjusted trading began 2024-06-10 (Monday). Sources:
# NVIDIA's 2024 Stock Split Investor FAQ and its SEC Form 8-K of 2024-05-22.
_SPLIT_FRIDAY = date(2024, 6, 7)
_SPLIT_MONDAY = date(2024, 6, 10)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _next_business_day(d: date) -> date:
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d += timedelta(days=1)
    return d


def _probe_window(symbol: str, probe_date: date):
    """A 5-business-day window starting at/after probe_date, avoiding the
    late-May/June-19/July-4 holiday cluster by construction (callers pick
    probe_date accordingly). Returns (start_date, end_date) as date objects.
    """
    start = _next_business_day(probe_date)
    end = start
    added = 0
    while added < 4:
        end += timedelta(days=1)
        if end.weekday() < 5:
            added += 1
    return start, end


def _probe_has_data(symbol: str, session, probe_date: date) -> tuple[bool, date, date]:
    """Probe a single 5-business-day window starting at/after probe_date.

    A raised IntradayDataGapError is treated directly as "no data" at
    exactly the probed window, with no retry on a shifted window -- what
    correct bisection requires: attributing a LATER window's data-presence
    back to the ORIGINAL probe_date would corrupt the bisection in
    _measure_retention_depth (a later date having data does not prove the
    probed date does, and near the true boundary that would produce a
    falsely-early "data starts here" conclusion). A market holiday inside
    the probe week does not raise a spurious gap: fetch_intraday_bars
    uses a real NYSE trading calendar internally (see
    src/mining/intraday_data._trading_sessions).
    """
    start, end = _probe_window(symbol, probe_date)
    try:
        df = fetch_intraday_bars(
            symbol, start.isoformat(), end.isoformat(),
            cache_dir=DEFAULT_CACHE_DIR, session=session,
        )
        return (not df.empty), start, end
    except IntradayDataGapError:
        return False, start, end


def _bisect_retention_boundary(lo: date, hi: date, has_data_fn, on_probe=None) -> tuple[date, date, int]:
    """Bisect [lo, hi] down to a bracket spanning 8 days or fewer.

    `lo` must already be confirmed to have no data and `hi` to have data
    (the caller establishes and verifies both before calling this). The
    only side-effecting call this function makes is `has_data_fn(probe_date)
    -> bool`, evaluated at the actual bisection point on every round and
    never substituted with a different date's result. Kept
    side-effect-pure otherwise (besides the optional `on_probe` logging
    callback) so the algorithm itself is directly unit-testable against a
    fake `has_data_fn`, decoupled from any real network call -- see
    src/tests/test_validate_alpaca_intraday.py.
    """
    lo_bound, hi_bound = lo, hi
    iterations = 0
    while (hi_bound - lo_bound).days > 8:
        iterations += 1
        mid = lo_bound + (hi_bound - lo_bound) // 2
        result = has_data_fn(mid)
        if on_probe is not None:
            on_probe(iterations, mid, result)
        if result:
            hi_bound = mid
        else:
            lo_bound = mid
    return lo_bound, hi_bound, iterations


def _measure_retention_depth(symbol: str, session) -> dict:
    _log(f"\n[{symbol}] measuring SIP historical retention depth ...")

    # Establish outer bracket. hi is assumed to have data (recent); lo is
    # assumed to have none (multi-decade back). Both assumptions are
    # verified directly, and pushed outward if wrong, rather than trusted.
    hi = date(2025, 6, 16)
    lo = date(2005, 6, 15)

    hi_ok, hi_start, hi_end = _probe_has_data(symbol, session, hi)
    if not hi_ok:
        raise RuntimeError(
            f"[{symbol}] recent probe window {hi_start}..{hi_end} unexpectedly "
            "returned no data -- cannot establish a valid 'has data' bracket end"
        )
    _log(f"  recent probe {hi_start}..{hi_end}: DATA PRESENT (confirms upper bracket)")

    lo_ok, lo_start, lo_end = _probe_has_data(symbol, session, lo)
    while lo_ok:
        _log(f"  old probe {lo_start}..{lo_end}: DATA PRESENT (pushing lower bracket further back)")
        lo = date(lo.year - 10, lo.month, lo.day)
        lo_ok, lo_start, lo_end = _probe_has_data(symbol, session, lo)
    _log(f"  old probe {lo_start}..{lo_end}: no data (confirms lower bracket)")

    def _has_data(probe_date: date) -> bool:
        return _probe_has_data(symbol, session, probe_date)[0]

    def _on_probe(round_no: int, probe_date: date, result: bool) -> None:
        start, end = _probe_window(symbol, probe_date)
        verdict = "DATA PRESENT" if result else "no data"
        arrow = "narrowing hi" if result else "narrowing lo"
        _log(f"  bisect #{round_no}: probe {start}..{end}: {verdict} -> {arrow}")

    lo_bound, hi_bound, iterations = _bisect_retention_boundary(lo, hi, _has_data, _on_probe)

    _log(
        f"  [{symbol}] boundary bracketed: no confirmed data at/before probe "
        f"date {lo_bound.isoformat()}; data confirmed present at/after probe "
        f"date {hi_bound.isoformat()} ({iterations} bisection rounds)"
    )
    return {
        "symbol": symbol,
        "no_data_probe_date": lo_bound.isoformat(),
        "data_present_probe_date": hi_bound.isoformat(),
        "bisection_rounds": iterations,
    }


def _measure_adjustment_behavior(session) -> dict:
    _log(f"\n[{_ADJUSTMENT_SYMBOL}] measuring split-adjustment behavior across "
         f"the real {_ADJUSTMENT_SYMBOL} 10-for-1 split ({_SPLIT_FRIDAY} -> {_SPLIT_MONDAY}) ...")

    results = {}
    for adjustment in ("raw", "split"):
        # Fresh subdirectory per adjustment value. fetch_intraday_bars
        # already keys its own cache by (symbol, date, adjustment), so two
        # different adjustment values for the same symbol/date window cannot
        # collide even in the same cache_dir; the separate per-adjustment
        # cache_dir is extra (belt-and-suspenders) isolation for this
        # validation script only, not load-bearing.
        cache_dir = os.path.join(DEFAULT_CACHE_DIR, f"_validation_{adjustment}")
        df = fetch_intraday_bars(
            _ADJUSTMENT_SYMBOL, "2024-06-05", "2024-06-11",
            cache_dir=cache_dir, session=session, adjustment=adjustment,
        )
        friday = df[df.index.date == _SPLIT_FRIDAY]
        monday = df[df.index.date == _SPLIT_MONDAY]
        if friday.empty or monday.empty:
            raise RuntimeError(
                f"[{_ADJUSTMENT_SYMBOL}] adjustment={adjustment}: missing bars on "
                f"{_SPLIT_FRIDAY} or {_SPLIT_MONDAY} -- cannot compare"
            )
        friday_close = float(friday["Close"].iloc[-1])
        monday_open = float(monday["Open"].iloc[0])
        ratio = friday_close / monday_open
        _log(
            f"  adjustment={adjustment!r}: {_SPLIT_FRIDAY} close={friday_close:.4f}, "
            f"{_SPLIT_MONDAY} open={monday_open:.4f}, ratio={ratio:.4f}"
        )
        results[adjustment] = {
            "friday_close": friday_close,
            "monday_open": monday_open,
            "ratio": ratio,
        }
    return results


def _write_findings(retention_results: list, adjustment_results: dict, recommended_adjustment: str) -> None:
    lines = []
    lines.append(f"# Alpaca Intraday Data Validation -- {date.today().isoformat()}")
    lines.append("")
    lines.append(
        "Real, live Alpaca SIP historical-bars API calls made via "
        "`scripts/validate_alpaca_intraday.py`, using the real "
        "`src/mining/intraday_data.fetch_intraday_bars` client and real "
        "`ALPACA_API_KEY_ID`/`ALPACA_API_SECRET_KEY` credentials from the "
        "environment. Not part of pytest; no fabricated numbers below -- "
        "every figure here was returned by the live API on the date this "
        "doc was written."
    )
    lines.append("")
    lines.append("## 1. SIP historical retention depth (QQQ, TQQQ)")
    lines.append("")
    lines.append(
        "Method: bisection search on a 5-business-day probe window. Each "
        "probe fetches that window through the real client; a raised "
        "`IntradayDataGapError` (zero RTH bars returned) means \"no data\" "
        "for that probe; a non-empty result means \"data present\". The "
        "bracket below is narrowed until it spans 8 days or fewer -- it is "
        "NOT claimed as an exact single boundary date, since a probe window "
        "spans multiple days and holidays/gaps inside a real window are "
        "possible even near the true boundary."
    )
    lines.append("")
    for result in retention_results:
        lines.append(f"### {result['symbol']}")
        lines.append("")
        lines.append(
            f"- No data confirmed at/before probe date **{result['no_data_probe_date']}**"
        )
        lines.append(
            f"- Data confirmed present at/after probe date **{result['data_present_probe_date']}**"
        )
        lines.append(f"- Bisection rounds: {result['bisection_rounds']}")
        lines.append("")
    lines.append("## 2. Split-adjustment behavior (NVDA 10-for-1 split)")
    lines.append("")
    lines.append(
        "Real split, verified before use (not guessed): NVIDIA announced a "
        "10-for-1 forward stock split 2024-05-22; effective after market "
        "close 2024-06-07; trading began on a split-adjusted basis at "
        "market open Monday 2024-06-10 (source: NVIDIA's own 2024 Stock "
        "Split Investor FAQ, and NVIDIA's Form 8-K filed with the SEC "
        "2024-05-22)."
    )
    lines.append("")
    lines.append(
        "Method: fetch NVDA minute bars for 2024-06-05..2024-06-11 twice -- "
        "once with `adjustment=\"raw\"`, once with `adjustment=\"split\"` -- "
        "and compare the Friday 2024-06-07 close to the Monday 2024-06-10 "
        "open. If Alpaca's `raw` value is truly unadjusted, that ratio "
        "should show the ~10x pre/post-split discontinuity; `split` should "
        "not."
    )
    lines.append("")
    lines.append("| adjustment | Fri 2024-06-07 close | Mon 2024-06-10 open | ratio |")
    lines.append("|---|---|---|---|")
    for adjustment, values in adjustment_results.items():
        lines.append(
            f"| {adjustment} | {values['friday_close']:.4f} | "
            f"{values['monday_open']:.4f} | {values['ratio']:.4f} |"
        )
    lines.append("")
    lines.append(f"**Conclusion:** recommended `fetch_intraday_bars` default: `adjustment=\"{recommended_adjustment}\"`.")
    lines.append("")

    os.makedirs(os.path.dirname(_FINDINGS_PATH), exist_ok=True)
    with open(_FINDINGS_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    _log(f"\nFindings written to {_FINDINGS_PATH}")


def main() -> int:
    key_id = os.environ.get("ALPACA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key_id or not secret:
        _log("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY not set in environment -- aborting.")
        return 2

    session = requests.Session()

    retention_results = [_measure_retention_depth(symbol, session) for symbol in _RETENTION_SYMBOLS]
    adjustment_results = _measure_adjustment_behavior(session)

    raw_ratio = adjustment_results["raw"]["ratio"]
    split_ratio = adjustment_results["split"]["ratio"]
    # A real ~10x discontinuity (allow 8x-12x for intraday noise) on `raw`
    # but not on `split` confirms `raw` is truly unadjusted and `split` is
    # the correct default for a client that will hold positions across
    # corporate actions. If neither shows the expected pattern, do not
    # guess -- surface it loudly instead of picking a default.
    raw_shows_discontinuity = 8.0 <= raw_ratio <= 12.0
    split_is_continuous = 0.8 <= split_ratio <= 1.2
    if raw_shows_discontinuity and split_is_continuous:
        recommended = "split"
    elif not raw_shows_discontinuity and split_is_continuous:
        # raw already looks adjusted (no discontinuity) -- split is still
        # the safe, explicit choice for a position-holding client.
        recommended = "split"
    else:
        _log(
            "\nWARNING: observed raw/split ratios did not match either expected "
            "pattern cleanly -- no default is recommended. Inspect the findings "
            "doc and set src/mining/intraday_data.py's DEFAULT_ADJUSTMENT by hand."
        )
        recommended = "UNRESOLVED -- see findings doc, manual review required"

    _write_findings(retention_results, adjustment_results, recommended)
    _log(f"\nRecommended adjustment default: {recommended}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
