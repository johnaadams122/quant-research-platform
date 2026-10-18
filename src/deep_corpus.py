"""Deep index-OWNER corpora for the Nasdaq-100 and S&P 500.

WHY THIS EXISTS
---------------
The QQQ/SPY ETF series are poor proxies for their own indices in early eras:
their annualized daily tracking error against the official index is several
times larger in the early years than after 2010.

The dot-com window -- the exact regime this project most wants the HMM to see --
is where QQQ is worst (ETF premium/discount before the create/redeem mechanism
matured, with single days where the ETF and the index moved by very different
amounts). This module sources the index levels from the OWNER instead, and
extends history:

    NDX_TR  1985-01-31 -> today   (causal eval from ~1989-01, incl. Black Monday)
    SPX_TR  1988-01-04 -> today   (causal eval from ~1991-12; no 1987 -- the
                                   total-return index has no back-history before
                                   1988: it is base-set to ^GSPC at
                                   1987-12-31.)

CONSTRUCTION
------------
NDX_TR : XNDX (official total return) from 1999-03-04; before that, NDX price
         return chained backward. Justified -- the realized NDX dividend
         yield around 1999-2001 was a small fraction of one percent and QQQ
         paid no distribution until late 2003, so price ~= total return
         pre-2003 (the error is well under the CAGR gap between a synthetic
         and a real 3x series). XNDX is base-set to NDX at inception,
         so the splice scale is 1.
SPX_TR : ^SP500TR directly (already total return). Pre-1988 is deliberately NOT
         reconstructed -- out of scope.

DIA is deliberately absent: daily DJIA total return before 1998 is not
obtainable (owner licence-gated; free redistributors disagree materially on
the same day), and splicing a price-return leg onto the total-return DIA leg would put a
silent dividend discontinuity at 1998 -- biasing exactly the leveraged-vs-1x
comparison this project tests.

DATA TERMS: this module fetches index levels at run time and the repository
ships none of that data. Anyone running it is responsible for complying with
each data owner's own terms of use.

TRAPS GUARDED HERE (each one silently corrupts the series; do not remove)
-----------------------------------------------------------
1. Nasdaq rows carry the level in "Value" -- "Close" is null on EVERY row.
2. Nasdaq returns rows NEWEST-FIRST.
3. Nasdaq emits Value=0.0 PLACEHOLDER rows for market closures (2012-10-29,
   Hurricane Sandy). log(0) -> -inf silently NaNs every downstream statistic,
   even when only a single such row is present.
4. A near-empty HTTP 200 is the documented silent-failure mode for index
   endpoints (Yahoo ^DJITR returns 200 + error:null + exactly 1 bar). Assert
   row counts; never trust the status code.
5. Yahoo range="max" silently returns monthly/quarterly bars while still
   labelling interval=1d -- always pass an explicit start date.
"""
import argparse
import datetime as dt
import os
import re

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from data_pipeline import OUTPUT_DIR, build_synthetic_3x, fetch_risk_free_rate, save_to_csv

NASDAQ_URL = "https://indexes.nasdaq.com/Index/HistoryData"
NASDAQ_HDRS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Content-Type": "application/json",
    "X-Requested-With": "XMLHttpRequest",
    "Origin": "https://indexes.nasdaq.com",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}

# Synthetic-3x cost parity with the existing QQQ/SPY configs (same real ETFs).
DEEP_SYNTHETIC_CONFIGS = [
    {"underlying": "NDX_TR", "real_etf": "TQQQ", "leverage": 3.0,
     "annual_expense_ratio": 0.0088, "financing_spread": 0.005,
     "dividend_yield": 0.0, "trading_days": 252},   # XNDX is already total return
    {"underlying": "SPX_TR", "real_etf": "SPXL", "leverage": 3.0,
     "annual_expense_ratio": 0.0091, "financing_spread": 0.005,
     "dividend_yield": 0.0, "trading_days": 252},   # ^SP500TR is already total return
]


def _drop_nonpositive(s, label):
    """Trap 3: closure placeholders / bad ticks would become -inf under log()."""
    bad = s[~(s > 0)]
    if len(bad):
        print(f"  {label}: dropped {len(bad)} non-positive row(s): "
              f"{[d.date().isoformat() for d in bad.index]}")
        s = s[s > 0]
    return s


def fetch_nasdaq_index(idx, min_rows, end_date=None):
    """Official Nasdaq Global Index Watch history -> Series[Timestamp, level]."""
    end = end_date or dt.date.today().isoformat()
    body = {"id": idx, "timeOfDay": "EOD",
            "startDate": "1980-01-01T00:00:00.000",
            "endDate": f"{end}T00:00:00.000", "timeFrame": "custom"}
    h = dict(NASDAQ_HDRS)
    h["Referer"] = f"https://indexes.nasdaq.com/Index/History/{idx}"
    r = requests.post(NASDAQ_URL, json=body, headers=h, timeout=120)
    r.raise_for_status()
    rows = r.json().get("aaData") or []
    assert len(rows) >= min_rows, (                       # trap 4
        f"{idx}: {len(rows)} rows < required {min_rows} -- silent-failure trap")
    recs = []
    for row in rows:
        m = re.search(r"/Date\((-?\d+)\)/", str(row.get("TimeStamp", "")))
        v = row.get("Value")                              # trap 1: NOT "Close"
        if m and v is not None:
            d = dt.datetime(1970, 1, 1) + dt.timedelta(milliseconds=int(m.group(1)))
            recs.append((d.date(), float(v)))
    s = pd.Series(dict(recs)).sort_index()                # trap 2: newest-first
    s.index = pd.to_datetime(s.index)
    s = _drop_nonpositive(s, idx)                         # trap 3
    assert s.index.is_monotonic_increasing and s.index.is_unique
    return s


def fetch_yahoo_index(ticker, start, min_rows):
    """Yahoo index history with an explicit start (trap 5) + row assertion."""
    df = yf.download(ticker, start=start, progress=False,
                     auto_adjust=False, threads=False)
    s = df["Close"].squeeze().dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    assert len(s) >= min_rows, f"{ticker}: {len(s)} rows < required {min_rows}"
    return _drop_nonpositive(s, ticker)


def build_ndx_tr():
    """XNDX total return, extended back to 1985 with NDX price return."""
    ndx = fetch_nasdaq_index("NDX", 10000)
    xndx = fetch_nasdaq_index("XNDX", 6000)
    assert ndx.index[0] <= pd.Timestamp("1985-02-01"), "NDX start later than expected"
    assert abs(ndx.iloc[0] - 125.0) < 1e-6, f"NDX base != 125.0 (got {ndx.iloc[0]})"
    cut = xndx.index[0]
    scale = xndx.iloc[0] / ndx[ndx.index <= cut].iloc[-1]
    deep = pd.concat([ndx[ndx.index < cut] * scale, xndx]).sort_index()
    deep = deep[~deep.index.duplicated(keep="last")]
    print(f"  NDX_TR: spliced at {cut.date()} scale={scale:.10f} -> {len(deep)} rows")
    return deep


def build_spx_tr():
    """^SP500TR is already total return; pre-1988 reconstruction is out of scope."""
    s = fetch_yahoo_index("^SP500TR", "1980-01-01", 9000)
    assert s.index[0] <= pd.Timestamp("1988-01-05"), "SP500TR start later than expected"
    print(f"  SPX_TR: {len(s)} rows from {s.index[0].date()}")
    return s


def validate(name, s, expect_start, expect_moves):
    """Self-verification: no non-finite returns, sane calendar, real crashes."""
    r = np.log(s).diff().replace([np.inf, -np.inf], np.nan).dropna()
    assert len(r) == len(s) - 1, f"{name}: non-finite returns survived"
    assert s.index[0] <= pd.Timestamp(expect_start), f"{name}: start regressed"
    obs = r.groupby(r.index.year).size()
    mid = obs[(obs.index > s.index[0].year) & (obs.index < r.index[-1].year)]
    assert mid.between(240, 260).all(), f"{name}: calendar anomaly\n{mid[~mid.between(240,260)]}"
    print(f"  {name}: {len(s)} rows {s.index[0].date()}->{s.index[-1].date()} | "
          f"sigma_d {r.std():.4%} | obs/yr {mid.min()}-{mid.max()}")
    for d, lo, hi in expect_moves:
        v = r.get(pd.Timestamp(d))
        assert v is not None and lo <= v <= hi, f"{name}: {d} move {v} outside [{lo},{hi}]"
        print(f"    verified {d}: {v:+.2%}")
    if len(s) > 1000:
        print(f"    causal eval starts ~{s.index[1000].date()} (1000-bar warmup)")


def main():
    ap = argparse.ArgumentParser(description="Build deep index-owner corpora")
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    args = ap.parse_args()

    print("Building deep corpora (index-owner sourced)")
    series = {"NDX_TR": build_ndx_tr(), "SPX_TR": build_spx_tr()}

    print("\nValidating:")
    validate("NDX_TR", series["NDX_TR"], "1985-02-01",
             [("1987-10-19", -0.20, -0.12)])            # Black Monday
    validate("SPX_TR", series["SPX_TR"], "1988-01-05",
             [("2020-03-16", -0.15, -0.10)])            # COVID

    print("\nWriting underlyings + synthetic 3x legs:")
    rf = fetch_risk_free_rate("max")
    for cfg in DEEP_SYNTHETIC_CONFIGS:
        tk = cfg["underlying"]
        df = series[tk].rename(tk).reset_index()
        df.columns = ["Date", tk]
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        save_to_csv(df, tk, args.output_dir)
        syn = build_synthetic_3x(
            df, tk, leverage=cfg["leverage"],
            annual_expense_ratio=cfg["annual_expense_ratio"],
            annual_financing_rate=rf, financing_spread=cfg["financing_spread"],
            dividend_yield=cfg["dividend_yield"], trading_days=cfg["trading_days"])
        save_to_csv(syn, f"{tk}_3x_synthetic", args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
