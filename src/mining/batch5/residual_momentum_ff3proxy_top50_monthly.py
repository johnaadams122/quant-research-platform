"""Candidate: residual_momentum_ff3proxy_top50_monthly (bespoke driver;
run only via scripts/run_residual_momentum_overlay_abi1.py, never the generic
scripts/run_abi1_mining.py dispatch -- see "Dispatch safety").

PRE-REGISTERED RULE. Every constant below was fixed ex-ante and none was tuned
after any result. The regression, formation and skip lengths and the score are
from the literature; N (about a decile), the overlay notional (the Reg-T bound),
the placebo seed and draw count and the 0.95 percentile bar are design choices,
not literature values::

    UNIVERSE  the project's cached US large-cap panel (S&P 500 + Nasdaq-100
              constituents scraped 2026-08-29), minus two documented vendor
              defects, minus the duplicate share classes, minus any name that
              fails the panel hygiene checks below
    DECISION  the close of the last exchange session of month M
    SIGNAL    for every name with a full 36 monthly returns (months M-35..M):
              OLS  R_i = a + b1*MKT + b2*SMB + b3*HML + e   with the tradeable
              proxies MKT = SPY, SMB = IWM - SPY, HML = IVE - IVW (monthly
              total returns from month-end adjusted closes; raw returns, no
              risk-free subtraction);
              S_i = sum of e over months M-11..M-1 (month M is skipped) divided
              by the sample standard deviation of those same 11 residuals
    COHORT    the 50 highest S_i (ties by ticker ascending; scores within a
              relative 1e-12 are one tie: identical histories can score a last
              bit apart, see ``cohort_overlay.select_top_n``)
    ENTRY     the OPEN of the first exchange session of month M+1
    EXIT      the CLOSE of the last exchange session of month M+1
    ACCOUNT   benchmark_plus_overlay: SPY held at all times PLUS 0.5 of equity
              times (equal-weight cohort - equal-weight of ALL eligible names)

Source: Blitz, Huij and Martens (2011), "Residual momentum", Journal of
Empirical Finance 18(3):506-521 (36-month rolling three-factor regression; the
sum of residuals over months t-12..t-2 scaled by their standard deviation, as
restated in Hanauer and Windmuller's later work). Also Blitz, Hanauer and
Vidojevic, "The idiosyncratic momentum anomaly" (International Review of
Economics and Finance, 2020). The cited evidence is a zero-investment top-minus-
bottom decile LONG-SHORT result over 1926-2009 and, for the later paper, through
2015 in US data; a long-only top-50 leg over a cap-weighted benchmark is NOT
tested in either source, and the headline advantage over raw momentum comes
largely from avoiding the 2009 momentum crash, a short-leg phenomenon.

Why the account is a benchmark-plus-overlay with a CONTROL short leg
--------------------------------------------------------------------
The panel is TODAY's index members (survivorship). A zero-signal equal-weight
basket of such a panel can earn a large in-sample excess over SPY with no signal
at all, purely from survivorship, while a point-in-time equal-weight index fund
would not. A whole-account top-50 test therefore cannot discriminate residual
momentum from universe look-ahead, and its drawdown would trip ABI-1's absolute cap regardless of the signal. The scored account
is instead SPY plus the OVERLAY (cohort minus equal-weight of all eligible
names, ``mining.cohort_overlay``): because both legs hold the same survivors the
overlay cancels the LEVEL of the survivorship baseline only. Rank-conditional
survivorship (past residual-momentum winners that later failed, or were added to
the index later, are missing or present asymmetrically in the cohort) is NOT
cancelled, and the permutation placebo cannot test it: it draws its random
cohorts from the same eligible set, so it shares that limitation. A pass is
therefore INCONCLUSIVE, not confirmatory, and a cut is the informative outcome.
The short leg is a RESEARCH CONTROL, not an investable account. If this passes,
an investable long-only version needs its own design and its own trial. The
risk-basis rule applies (risk refusals on the overlay-only excess account).

Pre-registered design decisions
-------------------------------
1. Overlay account with costs on both legs, ex-ante scheduled windows: as above.
2. The permutation placebo is mandatory (500 draws, seeded) and the zero-signal
   equal-weight control's own ABI-1 numbers are printed beside the result.
3. Universe hygiene: the share classes GOOGL, FOX and NWSA are dropped (kept:
   GOOG, FOXA, NWS; rule: the class whose history starts first in the panel,
   ties by ticker ascending -- a property of the data, not of any return);
   HUBB (documented 1994 feed defect and missing sessions) and MNST (documented
   split-adjustment race) are excluded; any name with an interior gap against
   the exchange calendar since the first needed month-end, a stale history, or a
   bar on a closed day is excluded by ``panel_hygiene`` and listed; and the
   builder fails closed if any eligible name lacks the entry Open or a window
   Close (nothing is dropped, shifted or filled). The hygiene exclusions (an
   interior gap since 2000-05-31, a failed 2026 quote gate) are decided on
   FULL-SAMPLE or 2026 data and applied to all history: a data-quality rule,
   not a return-based one, that removes a name from every window including
   earlier ones where its data was fine; it applies equally to both legs, so
   the effect is second order, but it is a future-conditioned membership rule.
4. Month-end decision/entry/exit dates come from the full exchange-session
   calendar (``exchange_calendars`` XNYS), cross-checked against SPY's own
   dates (``prepare`` refuses a SPY or factor-ETF history that is not exactly
   the exchange's session set); never from a ticker's own next-bar test, and
   not from src/trading_calendar.py (2026-2028 only). Actual sessions are used,
   so the one closure in 2003-2026 that touches a window boundary (Jan 2 2007,
   announced Dec 27 2006, before the Dec 29 decision) moves that entry to the
   first ACTUAL session; every window shares one entry and one exit date by
   construction.
5. Disclosures the catalog entry also carries: the eligible set grows over
   time, so the top 50 is a larger share of it early than late, not a
   constant decile; the cited
   evidence is long-short; the forced monthly round trip skips
   the month-boundary overnight gap; 2026 quoted spreads are applied to 2003
   trades (a uniform pipeline convention; direction unmeasured, plausibly
   understating early-year costs).
6. The factor ETFs and SPY are loaded with retry and fail-closed and frozen to a
   dated cache by the driver.
Decisions where the source rule leaves room (each the more conservative option):
the eligible-name definition is a full 36 monthly returns with a close on every
month-end session (a missing month-end bar is NEVER forward-filled: it makes the
name ineligible until it rolls out of the window); a final month whose exit lies
after the data is EXCLUDED and reported, never closed early; the residual
standard deviation is the sample one (ddof=1: rank-invariant, a constant factor
across names); the overlay notional is 0.5 (Reg-T: gross 1 + 2x with half of it
initial margin needs x <= 0.5; not tuned, and the excess Sharpe is scale-free).

What is NOT the paper: raw returns (the risk-free leg is common to every name,
so rank-neutral to first order) instead of excess returns; tradeable ETF
proxies instead of the Fama-French factors (the HML proxy in particular tracks
the true factor only loosely, so value exposure is only partly removed); a long-only top-50 cohort (about a decile) instead of the
value-weighted decile spread; equal weights; a large-cap panel of survivors.

Look-ahead: the regression uses months M-35..M and the score sums residuals of
months M-11..M-1, all known at the close of month M; the factors are tradeable
prices known at that close; entry is the next session's Open. A test perturbs
every bar after the decision session and shows the score unchanged.

Dispatch safety
---------------
``SHAPE = "pooled_single_asset"`` is a contract LABEL only. ``make_signal_fn``
always raises, so the generic dispatch (which would score plain long positions)
turns this family into a clean compute error; the only legitimate entry point is
``scripts/run_residual_momentum_overlay_abi1.py``.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from mining import cohort_overlay as _co
from mining import long_overlay as _overlay
from mining.batch5 import treasury_month_end_index_demand_overlay as _cal
from mining.long_overlay import CalendarCoverageError  # noqa: F401  (re-exported for the driver)

FAMILY_ID = "residual_momentum_ff3proxy_top50_monthly"
SHAPE = "pooled_single_asset"      # contract label only -- see "Dispatch safety"

# The scraped constituent list, exactly as data/mining/universe_snapshot_2026-08-29.json
# (a test proves equality). TRADED below removes the documented defects and the
# duplicate share classes.
SNAPSHOT_TICKERS = [
    "A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK",
    "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM", "ALAB", "ALB", "ALGN", "ALL",
    "ALLE", "ALNY", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP", "AMT", "AMZN", "ANET", "AON",
    "AOS", "APA", "APD", "APH", "APO", "APP", "APTV", "ARE", "ARES", "ARM", "ASML", "ATO",
    "AVGO", "AVY", "AWK", "AXON", "AXP", "AZO", "BA", "BAC", "BALL", "BAX", "BBY", "BDX", "BEN",
    "BF-B", "BG", "BIIB", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BNY", "BR", "BRK-B", "BRO",
    "BSX", "BX", "BXP", "C", "CAH", "CARR", "CASY", "CAT", "CB", "CBOE", "CBRE", "CCEP", "CCI",
    "CCL", "CDNS", "CDW", "CEG", "CF", "CFG", "CHD", "CHRW", "CHTR", "CI", "CIEN", "CINF", "CL",
    "CLX", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COHR", "COIN", "COO",
    "COP", "COR", "COST", "CPAY", "CPRT", "CPT", "CRH", "CRL", "CRM", "CRWD", "CRWV", "CSCO",
    "CSGP", "CSX", "CTAS", "CTSH", "CTVA", "CVNA", "CVS", "CVX", "D", "DAL", "DASH", "DD",
    "DDOG", "DE", "DECK", "DELL", "DG", "DGX", "DHI", "DHR", "DIS", "DLR", "DLTR", "DOC", "DOV",
    "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN", "DXCM", "EBAY", "ECHO", "ECL", "ED", "EFX",
    "EG", "EIX", "EL", "ELV", "EME", "EMR", "EOG", "EQIX", "EQT", "ERIE", "ES", "ESS", "ETN",
    "ETR", "EVRG", "EW", "EXC", "EXE", "EXPD", "EXPE", "EXR", "F", "FANG", "FAST", "FCX", "FDS",
    "FDX", "FDXF", "FE", "FER", "FERG", "FFIV", "FICO", "FIS", "FISV", "FITB", "FIX", "FLEX",
    "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GD", "GDDY", "GE", "GEHC", "GEN", "GEV",
    "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS",
    "GWW", "HAL", "HAS", "HBAN", "HCA", "HD", "HIG", "HII", "HLT", "HON", "HONA", "HOOD", "HPE",
    "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBKR", "IBM", "ICE", "IDXX",
    "IEX", "IFF", "INCY", "INTC", "INTU", "INVH", "IP", "IQV", "IR", "IRM", "ISRG", "IT", "ITW",
    "IVZ", "J", "JBHT", "JBL", "JCI", "JKHY", "JNJ", "JPM", "KDP", "KEY", "KEYS", "KHC", "KIM",
    "KKR", "KLAC", "KMB", "KMI", "KO", "KR", "KVUE", "L", "LDOS", "LEN", "LH", "LHX", "LII",
    "LIN", "LITE", "LLY", "LMT", "LNT", "LOW", "LRCX", "LULU", "LUV", "LVS", "LYB", "LYV", "MA",
    "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MELI", "MET", "META",
    "MGM", "MKC", "MLM", "MMM", "MNST", "MO", "MOS", "MPC", "MPWR", "MRK", "MRNA", "MRSH",
    "MRVL", "MS", "MSCI", "MSFT", "MSI", "MSTR", "MTB", "MTD", "MU", "NBIS", "NCLH", "NDAQ",
    "NDSN", "NEE", "NEM", "NFLX", "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS",
    "NUE", "NVDA", "NVR", "NWS", "NWSA", "NXPI", "O", "ODFL", "OKE", "OMC", "ON", "ORCL",
    "ORLY", "OTIS", "OXY", "PANW", "PAYX", "PCAR", "PCG", "PDD", "PEG", "PEP", "PFE", "PFG",
    "PG", "PGR", "PH", "PHM", "PKG", "PLD", "PLTR", "PM", "PNC", "PNR", "PNW", "PODD", "PPG",
    "PPL", "PRU", "PSA", "PSKY", "PSX", "PTC", "PWR", "PYPL", "Q", "QCOM", "RCL", "RDDT", "REG",
    "REGN", "RF", "RJF", "RKLB", "RL", "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RTX", "RVTY",
    "SBAC", "SBUX", "SCHW", "SHOP", "SHW", "SJM", "SLB", "SMCI", "SNA", "SNDK", "SNPS", "SO",
    "SOLV", "SPCX", "SPG", "SPGI", "SRE", "STE", "STLD", "STT", "STX", "STZ", "SW", "SWK",
    "SWKS", "SYF", "SYK", "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TGT",
    "TJX", "TKO", "TMO", "TMUS", "TPL", "TPR", "TRGP", "TRI", "TRMB", "TROW", "TRV", "TSCO",
    "TSLA", "TSN", "TT", "TTD", "TTWO", "TXN", "TXT", "TYL", "UAL", "UBER", "UDR", "UHS",
    "ULTA", "UNH", "UNP", "UPS", "URI", "USB", "V", "VEEV", "VICI", "VLO", "VLTO", "VMC",
    "VMRK", "VRSK", "VRSN", "VRT", "VRTX", "VST", "VTR", "VTRS", "VZ", "WAB", "WAT", "WBD",
    "WDAY", "WDC", "WEC", "WELL", "WFC", "WM", "WMB", "WMT", "WRB", "WSM", "WST", "WTW", "WY",
    "WYNN", "XEL", "XOM", "XYL", "XYZ", "YUM", "ZBH", "ZBRA", "ZTS"
]

DOCUMENTED_DEFECTS = {
    "HUBB": "documented 1994 vendor feed defect and 121 missing sessions "
            "(scripts/run_expanded_universe_mining.py)",
    "MNST": "documented split-adjustment race inside the recent window",
}
# dropped -> kept. Rule: keep the class whose history starts first in the panel
# cache; a tie goes to the ticker that sorts first. GOOG and GOOGL both start on
# the same day (tie -> GOOG), FOXA starts a day before FOX, NWS and NWSA tie
# (-> NWS). A property of the data, chosen without reference to any return.
SHARE_CLASS_DUPLICATES = {"GOOGL": "GOOG", "FOX": "FOXA", "NWSA": "NWS"}

TRADED = [t for t in SNAPSHOT_TICKERS
          if t not in DOCUMENTED_DEFECTS and t not in SHARE_CLASS_DUPLICATES]
FACTOR_SYMBOLS = ("IWM", "IVE", "IVW")
SIGNAL = list(FACTOR_SYMBOLS)      # signal-only; SPY (MKT and the core) is BENCHMARK
SNAPSHOTS: dict = {}
BENCHMARK = "SPY"

# Pre-registered constants (literature; nothing tuned)
REGRESSION_MONTHS = 36
FORMATION_MONTHS = 11
SKIP_MONTHS = 1
N_SELECT = 50
OVERLAY_NOTIONAL = 0.5
PLACEBO_DRAWS = 500
PLACEBO_SEED = 1729              # an arbitrary fixed seed (never tuned)
# A financed session may read a bill print at most this many calendar days old
# (a holiday weekend is 4-5); older means a hole in the bill series, which the
# shared ``causal_financing_rate`` deliberately carries across silently.
MAX_BILL_PRINT_AGE_DAYS = 10
FIRST_MONTH_END = "2000-05-31"     # first month-end close needed (IWM/IVE/IVW start 2000-05-26)

# Copied byte-for-byte into the catalog entry (data/mining/catalog.json).
PARAMETERS = {
    "universe": (
        "the project's cached US large-cap panel (current S&P 500 + "
        "Nasdaq-100 constituents; today's members, survivorship-biased) "
        "minus HUBB and MNST (documented vendor defects) minus the duplicate "
        "share classes GOOGL, FOX and NWSA minus any name failing the "
        "interior-gap, stale-history, closed-day or quote checks"
    ),
    "factor_proxies": "MKT = SPY; SMB-proxy = IWM minus SPY; HML-proxy = IVE minus IVW; monthly total returns from month-end adjusted closes on the exchange session calendar",
    "regression_window_months": REGRESSION_MONTHS,
    "regression": "OLS with intercept on raw monthly returns (no risk-free subtraction), a full 36 monthly returns required",
    "formation_months": FORMATION_MONTHS,
    "skip_months": SKIP_MONTHS,
    "score": "sum of regression residuals over months M-11..M-1 divided by the sample standard deviation of those same 11 residuals",
    "select_top_n": N_SELECT,
    "tie_break": "ticker ascending",
    "rebalance": "monthly",
    "decision": "close of the last exchange session of month M",
    "entry": "open of the first exchange session of month M+1",
    "exit": "close of the last exchange session of month M+1",
    "calendar": "exchange_calendars XNYS actual sessions, cross-checked against SPY's own dates",
    "account": "benchmark_plus_overlay: SPY held at all times plus overlay_notional_fraction times (equal-weight selected cohort minus equal-weight of all eligible names); the short equal-weight leg is a research control, not an investable account",
    "overlay_notional_fraction": OVERLAY_NOTIONAL,
    "financing_rate": "^IRX 13-week bill coupon-equivalent yield of the last close strictly before each session, on the long leg's notional, actual calendar days over 365, no spread; no short rebate or borrow fee",
    "cost_model": "full round trip of the quoted spread on every name of both legs, charged on the exit day",
    "placebo_draws": PLACEBO_DRAWS,
    "placebo_seed": PLACEBO_SEED,
    "benchmark": BENCHMARK,
}
# Structural floor (calendar only): set below the number of complete monthly
# windows the panel's date span allows from June 2003 onward, the sibling
# convention.
EXPECTED_MIN_TRADES = 270


def make_signal_fn(*args, **kwargs):
    """Always raises -- this family has no per-bar signal. See "Dispatch safety":
    the only legitimate entry point is
    ``scripts/run_residual_momentum_overlay_abi1.py``."""
    raise RuntimeError(
        f"{FAMILY_ID}: this family must only be run through its own bespoke "
        "driver (scripts/run_residual_momentum_overlay_abi1.py), never the "
        "generic scripts/run_abi1_mining.py dispatch -- it has no per-bar "
        "signal, and the generic path would score plain long positions "
        "instead of the benchmark-plus-overlay account.")


# ----------------------------------------------------------------- calendar

def _stamp(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize()


def _month_groups(start: pd.Timestamp, end: pd.Timestamp):
    """(last, first) exchange session of every calendar month touched by
    ``[start, end]``, from the exchange calendar alone (never from data). The
    calendar is read to 70 days past ``end`` so the following month is whole."""
    first_day = pd.Timestamp(start.year, start.month, 1)
    sessions = _cal.exchange_sessions(first_day, (end + pd.Timedelta(days=70)).normalize())
    grouped = pd.Series(sessions, index=sessions).groupby(sessions.to_period("M"))
    return grouped.max(), grouped.min()


def month_schedule(first_decision, last_decision) -> list:
    """``(decision, entry, exit)`` for every month M whose last exchange session
    lies in ``[first_decision, last_decision]``: the decision is the last
    session of M, the entry the first session of M+1, the exit the last session
    of M+1 (actual sessions, from the calendar alone)."""
    start, end = _stamp(first_decision), _stamp(last_decision)
    first_day = pd.Timestamp(start.year, start.month, 1)
    sessions = _cal.exchange_sessions(first_day, (end + pd.Timedelta(days=70)).normalize())
    return _co.month_windows(sessions, start, end)


def month_end_sessions(first_month_end, last_bar) -> pd.DatetimeIndex:
    """The last exchange session of every month from the month of
    ``first_month_end`` through the last month whose month-end is on or before
    ``last_bar`` (a month still in progress at the last bar has no month-end)."""
    start, end = _stamp(first_month_end), _stamp(last_bar)
    last, _ = _month_groups(start, end)
    return pd.DatetimeIndex([d for d in last.values if start <= d <= end])


# ------------------------------------------------------------------ signal

def monthly_return_tables(spy_frame, factor_frames: Mapping, panel_frames: Mapping,
                          month_ends: pd.DatetimeIndex) -> tuple:
    """``(returns, factors)``: month-by-name simple returns from the Close AT
    each month-end session (NaN when the name has no bar on that exact session:
    NO forward fill, so a missing month-end bar kills that month's return and
    the next) and the month-by-3 factor table (MKT, SMB, HML)."""
    def closes(frame_by_name):
        return pd.DataFrame({t: pd.Series(f["Close"]).reindex(month_ends)
                             for t, f in frame_by_name.items()})

    def to_returns(table):
        return table / table.shift(1) - 1.0

    fac = closes({"SPY": spy_frame, **{s: factor_frames[s] for s in FACTOR_SYMBOLS}})
    if not np.isfinite(fac.to_numpy(dtype=float)).all():
        raise ValueError("SPY and the factor ETFs must have a Close on every month-end session")
    fr = to_returns(fac)
    factors = pd.DataFrame({"MKT": fr["SPY"], "SMB": fr["IWM"] - fr["SPY"], "HML": fr["IVE"] - fr["IVW"]})
    return to_returns(closes(panel_frames)), factors


def scores_at(returns: pd.DataFrame, factors: pd.DataFrame, row: int) -> pd.Series:
    """Residual-momentum score of every name at month-end row ``row`` (a
    position in the month-end table): OLS of the 36 monthly returns of months
    M-35..M on ``[1, MKT, SMB, HML]``, then the sum of the residuals of months
    M-11..M-1 (M itself skipped) over their sample standard deviation. Uses ONLY
    rows ``row-35..row``. NaN for a name without a full 36 finite returns, or
    whose residuals have no variance (an exact factor combination)."""
    scores = pd.Series(np.nan, index=returns.columns, dtype=float, name="residual_momentum")
    if row < REGRESSION_MONTHS or row >= len(returns):
        return scores
    window = slice(row - REGRESSION_MONTHS + 1, row + 1)
    x = factors.to_numpy(dtype=float)[window]
    if not np.isfinite(x).all():
        return scores
    y = returns.to_numpy(dtype=float)[window]
    valid = np.isfinite(y).all(axis=0)
    if not valid.any():
        return scores
    design = np.column_stack([np.ones(REGRESSION_MONTHS), x])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("the factor proxies are collinear over the regression window")
    beta, *_ = np.linalg.lstsq(design, y[:, valid], rcond=None)
    resid = y[:, valid] - design @ beta
    last = REGRESSION_MONTHS - SKIP_MONTHS                     # exclusive: month M is skipped
    formation = resid[last - FORMATION_MONTHS:last]            # months M-11..M-1
    spread = formation.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = formation.sum(axis=0) / spread
    ok = np.isfinite(raw) & (spread > 1e-12)
    values = np.full(int(valid.sum()), np.nan)
    values[ok] = raw[ok]
    scores.iloc[np.flatnonzero(valid)] = values
    return scores


# -------------------------------------------------------------- preparation

def validate_market_calendar(spy_frame, factor_frames: Mapping) -> pd.DatetimeIndex:
    """Refuse (``mining.long_overlay.CalendarCoverageError``: reason codes
    ``unequal_trailing_coverage``, ``calendar_mismatch``,
    ``missing_exchange_session``, ``bar_on_closed_day``) any SPY or factor-ETF
    history that is not, inside their common span, identical to each other AND
    to the exchange's actual session set, or that ends on different sessions.
    Returns the validated common calendar. Nothing is trimmed, patched or
    filled. The driver runs this BEFORE panel hygiene so a vendor hole in the
    market frames is named as such and never misread as a panel defect."""
    if set(factor_frames) != set(FACTOR_SYMBOLS):
        raise ValueError(f"factor_frames must be exactly the registered factor ETFs {list(FACTOR_SYMBOLS)}; "
                         f"got {sorted(map(str, factor_frames))}")
    idx = _overlay.common_calendar(spy_frame, {s: factor_frames[s] for s in FACTOR_SYMBOLS})
    _cal.validate_session_coverage(idx)
    return idx


def prepare(spy_frame, factor_frames: Mapping, panel_frames: Mapping, *,
            first_month_end=FIRST_MONTH_END, n_select: int = N_SELECT) -> dict:
    """Validate the calendars and plan every complete monthly window.

    ``spy_frame`` and ``factor_frames`` (exactly IWM, IVE, IVW) must be, inside
    their common span, identical to each other and to the EXCHANGE's session set
    and must end on the same session (``mining.long_overlay.CalendarCoverageError``
    with ``unequal_trailing_coverage`` / ``calendar_mismatch`` /
    ``missing_exchange_session`` / ``bar_on_closed_day``): a hole is refused,
    never trimmed, patched or filled. ``panel_frames`` are the hygienic panel
    (see ``panel_hygiene``); a name's eligibility at a decision is a full 36
    monthly returns with a finite score. Only windows whose exit is on or before
    the last bar are planned (``open_window`` reports one still open and
    ``not_started`` a decision whose entry lies after the data)."""
    idx = validate_market_calendar(spy_frame, factor_frames)
    last_bar = idx[-1]
    first_me = _stamp(first_month_end)
    month_ends = month_end_sessions(first_me, last_bar)
    if len(month_ends) == 0 or month_ends[0] != first_me:
        raise ValueError(f"first_month_end {first_me.date()} must be the last exchange session of its month")
    returns, factors = monthly_return_tables(spy_frame, factor_frames, panel_frames, month_ends)

    def score_fn(decision):
        return scores_at(returns, factors, month_ends.get_loc(_stamp(decision)))

    schedule = (month_schedule(month_ends[REGRESSION_MONTHS], month_ends[-1])
                if len(month_ends) > REGRESSION_MONTHS else [])
    complete = [(d, e, x) for d, e, x in schedule if x <= last_bar]
    tail = [(d, e, x) for d, e, x in schedule if x > last_bar]
    plan = _co.plan_cohort_windows(complete, score_fn, n_select)
    open_window = None
    if tail and tail[0][1] <= last_bar:
        open_window = {"decision": tail[0][0], "entry": tail[0][1], "scheduled_exit": tail[0][2]}
    not_started = None
    pending = [t for t in tail if t[1] > last_bar]
    if pending:
        not_started = {"decision": pending[0][0], "entry": pending[0][1], "scheduled_exit": pending[0][2]}
    windows = plan["windows"]
    return {
        "windows": windows, "skipped": plan["skipped"], "open_window": open_window,
        "not_started": not_started, "month_ends": month_ends, "returns": returns,
        "factors": factors, "score_fn": score_fn, "spy_frame": spy_frame, "idx": idx,
        "n_select": n_select,
        "first_decision": windows[0].decision if windows else None,
        "last_exit": windows[-1].exit if windows else None,
        "eligible_counts": [len(w.eligible) for w in windows],
    }


class BillPrintStaleError(ValueError):
    """The bill print a financed session reads is too old, or there is none: a
    hole in the ^IRX series that the shared ``causal_financing_rate`` would
    carry across without a word. A ``ValueError`` that also carries a
    machine-readable ``reason_code`` (``stale_bill_print``), ``max_age_days``
    (None when a financed session has no earlier print at all), ``session`` (the
    ISO date of the oldest-print session) and ``n_sessions_over_limit``."""

    def __init__(self, message: str, *, max_age_days, session, n_sessions_over_limit):
        super().__init__(message)
        self.reason_code = "stale_bill_print"
        self.max_age_days = max_age_days
        self.session = session
        self.n_sessions_over_limit = n_sessions_over_limit


def bill_print_age_report(irx_frame, session_index, windows) -> dict:
    """Report-only: the age, in calendar days, of the bill print each FINANCED
    session (every session of every holding window) reads, i.e. the session's
    distance to the last ^IRX print STRICTLY BEFORE it (the causal rule of
    ``mining.long_overlay.causal_financing_rate``, re-derived here on its own).
    ``max_age_days`` is None when some financed session has no earlier print."""
    closes = pd.Series(irx_frame["Close"], dtype=float)
    closes = closes[~closes.index.duplicated(keep="last")].sort_index().dropna()
    dates = pd.DatetimeIndex(closes.index)
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    dates = dates.normalize()
    sessions = pd.DatetimeIndex(session_index)
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    sessions = sessions.normalize()
    financed = np.zeros(len(sessions), dtype=bool)
    for w in windows:
        financed |= (sessions >= _stamp(w.entry)) & (sessions <= _stamp(w.exit))
    sess = sessions[financed]
    prior = dates.searchsorted(sess.values, side="left") - 1           # last print strictly before
    if (prior < 0).any():
        first = sess[np.flatnonzero(prior < 0)[0]]
        return {"max_age_days": None, "session": first.strftime("%Y-%m-%d"),
                "n_financed_sessions": int(len(sess)), "n_sessions_over_limit": int((prior < 0).sum())}
    ages = np.asarray((sess.values - dates.values[prior]) / np.timedelta64(1, "D"), dtype=float).astype(int)
    worst = int(np.argmax(ages))
    return {"max_age_days": int(ages[worst]), "session": sess[worst].strftime("%Y-%m-%d"),
            "n_financed_sessions": int(len(sess)),
            "n_sessions_over_limit": int((ages > MAX_BILL_PRINT_AGE_DAYS).sum())}


def _require_fresh_bill_print(prepared: dict, irx_frame) -> dict:
    report = bill_print_age_report(irx_frame, prepared["spy_frame"].index, prepared["windows"])
    age = report["max_age_days"]
    if age is None:
        raise BillPrintStaleError(
            f"no bill print exists before the financed session {report['session']} "
            f"({report['n_sessions_over_limit']} financed session(s) affected)",
            max_age_days=None, session=report["session"], n_sessions_over_limit=report["n_sessions_over_limit"])
    if age > MAX_BILL_PRINT_AGE_DAYS:
        raise BillPrintStaleError(
            f"the bill print read on {report['session']} is {age} calendar days old (limit "
            f"{MAX_BILL_PRINT_AGE_DAYS}); {report['n_sessions_over_limit']} financed session(s) over the limit "
            "-- a hole in the ^IRX series that the shared causal financing rate would carry across silently",
            max_age_days=age, session=report["session"], n_sessions_over_limit=report["n_sessions_over_limit"])
    return report


def _window_info(prepared: dict) -> dict:
    counts = prepared["eligible_counts"]
    return {
        "n_windows": len(prepared["windows"]),
        "n_months_skipped": len(prepared["skipped"]),
        "open_window_excluded": prepared["open_window"],
        "not_started": prepared["not_started"],
        "first_decision": prepared["first_decision"], "last_exit": prepared["last_exit"],
        "eligible_min": int(min(counts)) if counts else None,
        "eligible_median": float(np.median(counts)) if counts else None,
        "eligible_max": int(max(counts)) if counts else None,
    }


def build_curve(prepared: dict, panel_frames: Mapping, irx_frame, *, cost_bps_by_symbol: Mapping,
                spy_entry_cost_bps: float) -> dict:
    """The family's SCORED account curve, built with the PINNED overlay notional
    (a driver cannot run other constants than the catalog entry records): SPY at
    all times plus ``OVERLAY_NOTIONAL`` times (equal-weight cohort minus
    equal-weight of all eligible names) -- the short leg is a RESEARCH CONTROL,
    not an investable account (``mining.cohort_overlay``). ``irx_frame`` is the
    raw ^IRX frame (Close in percent), loaded WITHOUT the positivity gate."""
    if not prepared["windows"]:
        raise ValueError("no complete window to score (fewer than the minimum eligible names, or no data)")
    if not isinstance(irx_frame, pd.DataFrame) or "Close" not in irx_frame.columns:
        raise ValueError("irx_frame must be a DataFrame with a Close column")
    bill_report = _require_fresh_bill_print(prepared, irx_frame)
    spy = prepared["spy_frame"]
    financing = _overlay.causal_financing_rate(irx_frame["Close"], spy.index)
    curve = _co.build_cohort_overlay_curve(
        spy, panel_frames, prepared["windows"], cost_bps_by_symbol=cost_bps_by_symbol,
        spy_entry_cost_bps=spy_entry_cost_bps, financing_rate_annual=financing,
        overlay_notional=OVERLAY_NOTIONAL)
    curve["window_info"] = {**_window_info(prepared), "bill_print_max_age_days": bill_report["max_age_days"]}
    return curve


def build_control_curve(prepared: dict, panel_frames: Mapping, *, cost_bps_by_symbol: Mapping) -> dict:
    """REPORT-ONLY zero-signal control on the same windows: the whole account
    holds the equal-weight basket of every eligible name (the cohort ignored).
    A RESEARCH CONTROL, never scored to the ledger."""
    if not prepared["windows"]:
        raise ValueError("no complete window to score")
    return _co.build_equal_weight_control_curve(
        prepared["spy_frame"], panel_frames, prepared["windows"], cost_bps_by_symbol=cost_bps_by_symbol)


def build_daily_control_curve(prepared: dict, panel_frames: Mapping) -> dict:
    """REPORT-ONLY zero-signal control, gross of costs: the whole account holds
    the equal-weight basket of every eligible name REBALANCED EVERY SESSION (the
    construction closest to a 'no-signal equal-weight basket of the panel'). A RESEARCH CONTROL and an upper bound, never scored to the ledger."""
    if not prepared["windows"]:
        raise ValueError("no complete window to score")
    return _co.build_daily_rebalanced_control_curve(prepared["spy_frame"], panel_frames, prepared["windows"])


def run_placebo(prepared: dict, panel_frames: Mapping, irx_frame, *, cost_bps_by_symbol: Mapping,
                spy_entry_cost_bps: float, n_draws: int = PLACEBO_DRAWS, seed: int = PLACEBO_SEED) -> dict:
    """REPORT-ONLY permutation placebo (500 seeded draws by default): random
    cohorts of the real size from the same eligible set, same schedule, costs
    and financing; reports the real rule's percentile. The short leg is a
    RESEARCH CONTROL."""
    if not prepared["windows"]:
        raise ValueError("no complete window to score")
    _require_fresh_bill_print(prepared, irx_frame)
    financing = _overlay.causal_financing_rate(irx_frame["Close"], prepared["spy_frame"].index)
    return _co.permutation_placebo(
        prepared["spy_frame"], panel_frames, prepared["windows"], cost_bps_by_symbol=cost_bps_by_symbol,
        spy_entry_cost_bps=spy_entry_cost_bps, financing_rate_annual=financing,
        overlay_notional=OVERLAY_NOTIONAL, n_draws=n_draws, seed=seed)


# ---------------------------------------------------------------- hygiene

_MAX_LISTED = 5


def panel_hygiene(panel_frames: Mapping, *, calendar_index, first_needed=FIRST_MONTH_END) -> dict:
    """Exclude, and report by name and reason, every panel member whose history
    cannot be trusted; nothing is repaired. Against ``calendar_index`` (SPY's own
    sessions, already validated against the exchange):

    * ``documented_defect`` -- HUBB and MNST;
    * ``no_history`` -- no bar on or after ``first_needed``;
    * ``stale_history`` -- the last bar precedes the calendar's last session;
    * ``interior_gap`` -- a calendar session between the name's first in-scope
      bar and its last bar has no bar (a gap before ``first_needed`` is ignored);
    * ``bar_on_closed_day`` -- a bar on a date the calendar does not list.

    Returns ``{"kept": {ticker: frame}, "excluded": {ticker: {"reason": ...,
    ...}}}``; ``kept`` holds the caller's own frames unmodified."""
    calendar = pd.DatetimeIndex(calendar_index)
    if calendar.tz is not None:
        calendar = calendar.tz_localize(None)
    calendar = calendar.normalize()
    first = _stamp(first_needed)
    scope = calendar[calendar >= first]
    last_session = calendar[-1]
    kept, excluded = {}, {}
    for ticker, frame in panel_frames.items():
        if ticker in DOCUMENTED_DEFECTS:
            excluded[ticker] = {"reason": "documented_defect", "detail": DOCUMENTED_DEFECTS[ticker]}
            continue
        index = pd.DatetimeIndex(frame.index).normalize()
        in_scope = index[index >= first]
        if len(in_scope) == 0:
            excluded[ticker] = {"reason": "no_history"}
            continue
        last = in_scope[-1]
        if last < last_session:
            excluded[ticker] = {"reason": "stale_history", "last_bar": last.strftime("%Y-%m-%d"),
                                "last_session": last_session.strftime("%Y-%m-%d")}
            continue
        expected = scope[(scope >= index[0]) & (scope <= last)]   # scope already starts at first_needed
        missing = expected.difference(in_scope)
        if len(missing):
            excluded[ticker] = {"reason": "interior_gap", "n_missing": int(len(missing)),
                                "dates": [d.strftime("%Y-%m-%d") for d in missing[:_MAX_LISTED]]}
            continue
        extra = in_scope[in_scope <= last_session].difference(scope)
        if len(extra):
            excluded[ticker] = {"reason": "bar_on_closed_day", "n_extra": int(len(extra)),
                                "dates": [d.strftime("%Y-%m-%d") for d in extra[:_MAX_LISTED]]}
            continue
        kept[ticker] = frame
    return {"kept": kept, "excluded": excluded}
