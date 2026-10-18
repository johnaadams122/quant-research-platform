"""Pre-scoring vendor coverage check (report only; see mining.vendor_coverage).

    python scripts/check_vendor_coverage.py \\
        --symbols IEF,TLT,AGG,SPY,^IRX

Fetches each symbol EXACTLY the way the scoring drivers do
(``data_pipeline.fetch_ohlc(symbol, start="max")``), then lists, per symbol, the
actual NYSE sessions in the window with no bar, any bar on a closed day, and
whether the last bar is today's in-progress bar. Exit 0 = every symbol clean,
1 = at least one finding, 2 = a symbol could not be fetched. It never trims,
fills or repairs anything and writes no file.

``--suggest-cutoff`` appends one last line, ``last complete session: YYYY-MM-DD``
(the last actual session before the first hole, complete within the window
checked), ``none in window``, or ``unavailable`` when a symbol could not be
fetched. It is a suggestion only and never changes the exit code.

A bar is counted the way the scoring drivers count it. A row the vendor lists
with no prices at all (every OHLC field blank) is dropped first, exactly as
``run_batch5_mining.load_frame`` drops it, so that session reads as MISSING and
the suggested cutoff never lands on or after it. A row with only some prices is
kept, as the drivers keep it. A history with no row that has any price is a
failed fetch (exit 2), as it is for a driver.

A bond-market holiday on which a bill-rate series such as ^IRX has no bar shows
up as a MISSING session. The treasury driver tolerates that gap only when no
history end is set, and it refuses a cutoff on such a day, so the suggested
cutoff errs on the safe side.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _path in (SRC_ROOT,):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mining import vendor_coverage  # noqa: E402

DEFAULT_LOOKBACK_DAYS = 30


def _default_fetch(symbol: str) -> pd.DataFrame:
    import data_pipeline
    return data_pipeline.fetch_ohlc(symbol, start="max")


def main(argv=None, fetch=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--symbols", required=True,
                        help="comma-separated: every series the driver loads")
    parser.add_argument("--start", default=None,
                        help="window start YYYY-MM-DD (default: 30 days before --end)")
    parser.add_argument("--end", default=None,
                        help="window end YYYY-MM-DD (default: today, ET)")
    parser.add_argument("--as-of", default=None,
                        help="ET clock for the live-bar test (default: now)")
    parser.add_argument("--suggest-cutoff", action="store_true",
                        help="after the report, print the last complete session "
                             "(a suggestion only; the exit code is unchanged)")
    args = parser.parse_args(argv)
    fetch = fetch or _default_fetch

    now_et = (pd.Timestamp(args.as_of, tz="America/New_York") if args.as_of
              else pd.Timestamp.now(tz="America/New_York"))
    end = pd.Timestamp(args.end) if args.end else now_et.tz_localize(None).normalize()
    start = (pd.Timestamp(args.start) if args.start
             else end - pd.Timedelta(days=DEFAULT_LOOKBACK_DAYS))

    indices, failed = {}, []
    for symbol in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        try:
            frame = fetch(symbol)
            if frame is None or len(frame) == 0:
                raise RuntimeError("empty history")
            indices[symbol] = vendor_coverage.driver_index_from_frame(frame)
        except Exception as exc:   # report and continue: one bad symbol must not hide the rest
            failed.append((symbol, f"{type(exc).__name__}: {exc}"))
    if not indices:
        print("no symbol could be fetched: " + json.dumps(failed))
        if args.suggest_cutoff:
            print("last complete session: unavailable (no symbol fetched)")
        return 2
    report = vendor_coverage.coverage_report(indices, start=start, end=end,
                                             as_of=now_et)
    print(f"window {report['window'][0]} .. {report['window'][1]}   ok={report['ok']}")
    for symbol, row in report["symbols"].items():
        flags = []
        if row["missing"]:
            flags.append("MISSING " + ",".join(row["missing"]))
        if row["extra"]:
            flags.append("BAR-ON-CLOSED-DAY " + ",".join(row["extra"]))
        if row["live_partial_bar"]:
            flags.append("LIVE-PARTIAL-LAST-BAR")
        print(f"  {symbol:8s} last={row['last']} n={row['n_in_window']:3d}  "
              + ("; ".join(flags) if flags else "clean"))
    for date, names in report["missing_by_date"].items():
        print(f"  no bar on {date}: {', '.join(names)}")
    for symbol, why in failed:
        print(f"  FETCH FAILED {symbol}: {why}")
    if args.suggest_cutoff:
        if failed:   # a partial symbol set cannot vouch for a complete session
            print("last complete session: unavailable (a symbol failed to fetch)")
        else:
            cutoff = vendor_coverage.last_complete_session(report)
            print(f"last complete session: {cutoff or 'none in window'}")
    if failed:
        return 2
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
