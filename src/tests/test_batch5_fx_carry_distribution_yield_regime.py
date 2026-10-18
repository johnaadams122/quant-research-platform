"""Tests for src/mining/batch5/fx_carry_distribution_yield_regime.py.

Synthetic series only for arithmetic/behavior (no network, deterministic,
and no data file is read). The frozen catalog entry's parameters (hardcoded
below) are used ONLY to pin PARAMETERS byte-identity.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _path in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mining import backtest_engine  # noqa: E402
from mining.batch5 import fx_carry_distribution_yield_regime as mod  # noqa: E402

# The frozen catalog entry's "parameters" field, copied verbatim from the
# pre-registered catalog entry.
CATALOG_PARAMETERS = {
    "yield_lookback_months": 12,
    "regime_lookback_months": 36,
    "rebalance": "monthly",
    "long_only": True,
}


# ------------------------------------------------------------ contract

def test_contract_constants_match_the_frozen_catalog_entry():
    assert mod.FAMILY_ID == "fx_carry_distribution_yield_regime"
    assert mod.SHAPE == "single_long"
    assert mod.TRADED == ["FXA"]
    assert mod.SIGNAL == []
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "FXA"
    assert json.dumps(mod.PARAMETERS, sort_keys=True) == json.dumps(
        CATALOG_PARAMETERS, sort_keys=True)
    assert mod.YIELD_LOOKBACK_MONTHS == 12
    assert mod.REGIME_LOOKBACK_MONTHS == 36
    assert mod.EXPECTED_MIN_TRADES > 0


# ------------------------------------------------------------ fixtures

def _frame_from_close(close: pd.Series) -> pd.DataFrame:
    values = close.to_numpy(dtype=float)
    return pd.DataFrame({"Open": values, "High": values * 1.001,
                         "Low": values * 0.999, "Close": values},
                        index=close.index)


def _build_n_month_series(n_months: int, amounts, price: float = 100.0,
                          start: str = "2000-01-01", div_bday_index: int = 4):
    """One dividend event per month (fixed business-day position), constant
    ``price`` throughout. Returns (close, dividends, month_end_dates) where
    ``month_end_dates[i]`` is the actual last business day of month i
    (0-indexed)."""
    periods = pd.period_range(start, periods=n_months, freq="M")
    close_idx, div_dates, month_ends = [], [], []
    for i, period in enumerate(periods):
        bdays = pd.bdate_range(period.start_time, period.end_time)
        close_idx.extend(bdays)
        div_dates.append(bdays[div_bday_index])
        month_ends.append(bdays[-1])
    close = pd.Series(price, index=pd.DatetimeIndex(close_idx), dtype=float)
    dividends = pd.Series([float(a) for a in amounts],
                          index=pd.DatetimeIndex(div_dates), dtype=float)
    return close, dividends, month_ends


def _build_regime_series(n_months: int = 90, base_price: float = 100.0,
                         monthly_dividend: float = 1.0, drop_month_idx: int = 50,
                         drop_price: float = 60.0, start: str = "2000-01-01",
                         div_bday_index: int = 4):
    """90 months of constant monthly distributions (amount 1.0) and constant
    price 100.0, EXCEPT month ``drop_month_idx`` where price is cut to
    ``drop_price`` for that month only. Trailing-12-month distribution sum is
    a constant 12.0 everywhere once warmed up, so yield = 12/price: 0.12 at
    baseline, 0.2 during the single dropped month. See the module's own
    "Interpretation choices" for why this makes exactly one month cross its
    own trailing-36-month median (which stays anchored at 0.12, since only
    one value out of the 36-window is ever elevated)."""
    periods = pd.period_range(start, periods=n_months, freq="M")
    close_idx, close_vals, div_dates, month_ends = [], [], [], []
    for i, period in enumerate(periods):
        bdays = pd.bdate_range(period.start_time, period.end_time)
        price = drop_price if i == drop_month_idx else base_price
        close_vals.extend([price] * len(bdays))
        close_idx.extend(bdays)
        div_dates.append(bdays[div_bday_index])
        month_ends.append(bdays[-1])
    close = pd.Series(close_vals, index=pd.DatetimeIndex(close_idx), dtype=float)
    dividends = pd.Series(monthly_dividend, index=pd.DatetimeIndex(div_dates),
                          dtype=float)
    return close, dividends, month_ends


# ------------------------------------------------------ month_end_mask

def test_month_end_mask_flags_last_business_day_of_each_month():
    index = pd.bdate_range("2021-01-25", "2021-03-10")
    mask = mod.month_end_mask(index)
    flagged = set(d.strftime("%Y-%m-%d") for d in index[mask])
    # Jan 2021: 31st is a Sunday -> last bday is Fri 2021-01-29.
    # Feb 2021 (not a leap year): 28th is a Sunday -> last bday Fri 2021-02-26.
    assert flagged == {"2021-01-29", "2021-02-26"}
    # 2021-03-10 is a Wednesday; weekdays remain in March after it, so the
    # series' own final bar is correctly NOT flagged as month-end.
    assert not mask[-1]


def test_month_end_mask_flags_a_final_bar_that_truly_is_month_end():
    index = pd.bdate_range("2021-01-25", "2021-02-26")
    mask = mod.month_end_mask(index)
    assert mask[-1]
    assert index[-1] == pd.Timestamp("2021-02-26")


def test_month_end_mask_empty_index():
    assert mod.month_end_mask(pd.DatetimeIndex([])).size == 0


# ------------------------------------------------------- yield arithmetic

def test_monthly_trailing_yield_known_values_and_warmup():
    n = 14
    amounts = list(range(1, n + 1))          # month k (1-indexed) pays k
    close, dividends, month_ends = _build_n_month_series(n, amounts)
    yld = mod.monthly_trailing_yield(close, dividends, lookback_months=12)
    assert len(yld) == n
    assert list(yld.index) == month_ends
    # First 11 months (0-indexed 0..10): fewer than 12 calendar months of
    # history -> NaN, never an understated yield.
    assert yld.iloc[:11].isna().all()
    # sum(1..12)=78, sum(2..13)=90, sum(3..14)=102; price is constant 100.
    assert yld.iloc[11] == pytest.approx(78.0 / 100.0)
    assert yld.iloc[12] == pytest.approx(90.0 / 100.0)
    assert yld.iloc[13] == pytest.approx(102.0 / 100.0)


def test_monthly_trailing_yield_zero_distribution_month_is_zero_not_a_gap():
    """A calendar month with literally no distribution contributes 0 to the
    trailing sum -- a real economic fact for FXA (see module docstring), not
    a data error and never NaN by itself."""
    n = 14
    # Every month pays 1.0 EXCEPT month 10 (0-indexed), which has no event at
    # all (dividends are always > 0 when present, so a missing month is
    # represented by absence, never a zero-amount row).
    periods = pd.period_range("2000-01-01", periods=n, freq="M")
    close_idx, div_dates = [], []
    for i, period in enumerate(periods):
        bdays = pd.bdate_range(period.start_time, period.end_time)
        close_idx.extend(bdays)
        if i != 10:
            div_dates.append(bdays[4])
    close = pd.Series(100.0, index=pd.DatetimeIndex(close_idx), dtype=float)
    dividends = pd.Series(1.0, index=pd.DatetimeIndex(div_dates), dtype=float)
    yld = mod.monthly_trailing_yield(close, dividends, lookback_months=12)
    # trailing sum at month 11 (0-indexed) = months0..11 = 12 months, one
    # (month10) missing its event -> sum = 11.0, not NaN.
    assert yld.iloc[11] == pytest.approx(11.0 / 100.0)


def test_dividend_on_a_non_trading_date_counts_from_the_next_session_only():
    close = pd.Series(50.0, index=pd.bdate_range("2000-01-01", periods=20))
    saturday = pd.Timestamp("2000-01-08")             # a Saturday, not a session
    assert saturday not in close.index
    dividends = pd.Series([1.0], index=[saturday])
    on_cal = mod._distributions_on_calendar(dividends, close.index)
    next_session = close.index[close.index.searchsorted(saturday)]
    before = close.index[close.index.get_loc(next_session) - 1]
    assert on_cal.loc[before] == 0.0
    assert on_cal.loc[next_session] == pytest.approx(1.0)


# ------------------------------------------------------- regime median

def test_regime_median_requires_a_full_36_row_window_and_computes_correctly():
    values = [np.nan] * 4 + list(range(1, 39))   # 4 NaN rows + 38 defined
    idx = pd.bdate_range("2000-01-01", periods=len(values), freq="MS")
    yld = pd.Series(values, index=idx, dtype=float)
    median = mod.regime_median(yld, regime_lookback_months=36)
    assert median.iloc[:39].isna().all()          # position 39 is the first
    # position 39: window rows 4..39 = values 1..36 -> median 18.5
    assert median.iloc[39] == pytest.approx(18.5)
    # position 40: window rows 5..40 = values 2..37 -> median 19.5
    assert median.iloc[40] == pytest.approx(19.5)
    # position 41: window rows 6..41 = values 3..38 -> median 20.5
    assert median.iloc[41] == pytest.approx(20.5)


def test_regime_median_rejects_a_non_positive_window():
    with pytest.raises(ValueError):
        mod.regime_median(pd.Series([1.0, 2.0]), regime_lookback_months=0)


def test_monthly_trailing_yield_rejects_a_non_positive_lookback():
    close, dividends, _ = _build_n_month_series(3, [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        mod.monthly_trailing_yield(close, dividends, lookback_months=0)


# --------------------------------------------------------- decision rule

def test_standing_long_single_month_yield_spike_produces_exactly_one_long_span():
    close, dividends, month_ends = _build_regime_series()
    state = mod.standing_long(close, dividends)
    assert state.index.equals(close.index)
    assert state.dtype == bool
    start_pos = close.index.get_loc(month_ends[50])
    end_pos = close.index.get_loc(month_ends[51])   # exclusive
    expected_true = set(range(start_pos, end_pos))
    actual_true = set(np.flatnonzero(state.to_numpy()))
    assert actual_true == expected_true
    # nothing before the spike month and nothing after the following
    # month-end reassessment is ever long.
    assert not state.iloc[:start_pos].any()
    assert not state.iloc[end_pos:].any()


def test_standing_long_never_enters_when_yield_never_deviates_from_its_median():
    """A perfectly constant carry regime: yield always equals its own
    trailing median exactly -> the STRICT inequality never fires, so the
    family never goes long. This is the median-rule analogue of the sibling
    template's "zero rolling std yields no decision" guard."""
    close, dividends, _ = _build_regime_series(drop_month_idx=-1)  # no drop
    state = mod.standing_long(close, dividends)
    assert not state.any()


def test_signal_flags_are_the_exact_complement_of_standing_long():
    close, dividends, _ = _build_regime_series()
    flags = mod.signal_flags(close, dividends)
    assert list(flags.columns) == ["entry", "exit"]
    assert flags.index.equals(close.index)
    assert np.array_equal(flags["exit"].to_numpy(), ~flags["entry"].to_numpy())
    state = mod.standing_long(close, dividends)
    assert np.array_equal(flags["entry"].to_numpy(), state.to_numpy())


# ---------------------------------------------------------- look-ahead

def test_look_ahead_probe_perturbing_inputs_after_k_never_changes_flags_through_k():
    close, dividends, month_ends = _build_regime_series()
    k = close.index.get_loc(month_ends[60])          # well past the spike
    base = mod.signal_flags(close, dividends)
    close_p = close.copy()
    close_p.iloc[k + 1:] *= 0.3                       # huge future price crash
    extra = pd.Series(9.0, index=close.index[k + 1::5])
    dividends_p = pd.concat([dividends, extra]).sort_index()
    perturbed = mod.signal_flags(close_p, dividends_p)
    assert base.iloc[:k + 1].equals(perturbed.iloc[:k + 1])
    assert not base.iloc[k + 1:].equals(perturbed.iloc[k + 1:])


def test_holdout_blind_inputs_after_the_bucket_end_are_irrelevant():
    close, dividends, month_ends = _build_regime_series()
    frame = _frame_from_close(close)
    bucket = close.index[close.index.get_loc(month_ends[45]):
                         close.index.get_loc(month_ends[60])]
    inputs = {"FXA": {"unadjusted_close": close, "dividends": dividends}}
    signal_fn = mod.make_signal_fn("FXA", bucket, {"FXA": frame}, inputs=inputs)
    entries = np.array([signal_fn(frame.loc[:bucket[i]], i).entry
                        for i in range(len(bucket))])
    close_p = close.copy()
    close_p.iloc[close.index.get_loc(month_ends[60]):] = 1.0    # absurd future price
    dividends_p = pd.concat(
        [dividends, pd.Series(50.0, index=close.index[
            close.index.get_loc(month_ends[60])::3])]).sort_index()
    inputs_p = {"FXA": {"unadjusted_close": close_p, "dividends": dividends_p}}
    signal_fn_p = mod.make_signal_fn("FXA", bucket, {"FXA": frame}, inputs=inputs_p)
    entries_p = np.array([signal_fn_p(frame.loc[:bucket[i]], i).entry
                         for i in range(len(bucket))])
    assert np.array_equal(entries, entries_p)


# --------------------------------------------------------- end to end

def test_end_to_end_single_trade_from_the_spike_month():
    close, dividends, month_ends = _build_regime_series()
    frame = _frame_from_close(close)
    bucket_frame = frame.loc[month_ends[45]:].copy()
    inputs = {"FXA": {"unadjusted_close": close, "dividends": dividends}}
    signal_fn = mod.make_signal_fn("FXA", bucket_frame.index, {"FXA": frame},
                                   inputs=inputs)
    trades, in_market = backtest_engine.run_backtest(bucket_frame, signal_fn)
    assert len(trades) == 1
    entry_expected = close.index[close.index.get_loc(month_ends[50]) + 1]
    exit_expected = month_ends[51]
    assert trades["entry_date"].iloc[0] == entry_expected
    assert trades["exit_date"].iloc[0] == exit_expected
    assert trades["reason"].iloc[0] == "signal_exit"
    assert trades["entry_px"].iloc[0] == pytest.approx(100.0)
    assert trades["exit_px"].iloc[0] == pytest.approx(100.0)


def test_bucket_that_opens_inside_a_long_regime_enters_at_first_fill():
    close, dividends, month_ends = _build_regime_series()
    frame = _frame_from_close(close)
    mid_long = close.index.get_loc(month_ends[50]) + 3   # a few days into the long span
    bucket_frame = frame.iloc[mid_long:close.index.get_loc(month_ends[55])].copy()
    inputs = {"FXA": {"unadjusted_close": close, "dividends": dividends}}
    signal_fn = mod.make_signal_fn("FXA", bucket_frame.index, {"FXA": frame},
                                   inputs=inputs)
    trades, _ = backtest_engine.run_backtest(bucket_frame, signal_fn)
    assert len(trades) == 1
    assert trades["entry_date"].iloc[0] == bucket_frame.index[1]


def test_signal_fn_is_a_lookup_and_ignores_the_slice():
    close, dividends, month_ends = _build_regime_series()
    frame = _frame_from_close(close)
    bucket = close.index[close.index.get_loc(month_ends[45]):]
    inputs = {"FXA": {"unadjusted_close": close, "dividends": dividends}}
    signal_fn = mod.make_signal_fn("FXA", bucket, {"FXA": frame}, inputs=inputs)
    a = signal_fn(frame.loc[:bucket[50]], 50)
    b = signal_fn(None, 50)
    assert (a.entry, a.exit) == (b.entry, b.exit)
    with pytest.raises(IndexError):
        signal_fn(None, len(bucket))


# ------------------------------------------------------------ guards

def test_factory_rejects_the_wrong_ticker():
    close, dividends, _ = _build_regime_series(n_months=50)
    frame = _frame_from_close(close)
    inputs = {"FXA": {"unadjusted_close": close, "dividends": dividends}}
    with pytest.raises(ValueError, match="FXA"):
        mod.make_signal_fn("UUP", close.index[-10:], {"FXA": frame}, inputs=inputs)


def test_factory_fails_closed_when_the_data_file_does_not_cover_the_bucket():
    close, dividends, _ = _build_regime_series(n_months=50)
    frame = _frame_from_close(close)
    stale_close = close.iloc[:-30]
    inputs = {"FXA": {"unadjusted_close": stale_close, "dividends": dividends}}
    with pytest.raises(ValueError, match="ends"):
        mod.make_signal_fn("FXA", close.index[-10:], {"FXA": frame}, inputs=inputs)


def test_factory_fails_closed_on_an_empty_or_unsorted_bucket():
    close, dividends, _ = _build_regime_series(n_months=50)
    frame = _frame_from_close(close)
    inputs = {"FXA": {"unadjusted_close": close, "dividends": dividends}}
    with pytest.raises(ValueError, match="empty"):
        mod.make_signal_fn("FXA", pd.DatetimeIndex([]), {"FXA": frame}, inputs=inputs)
    backwards = close.index[-10:][::-1]
    with pytest.raises(ValueError, match="sorted"):
        mod.make_signal_fn("FXA", backwards, {"FXA": frame}, inputs=inputs)


def test_factory_fails_closed_on_a_missing_or_empty_frame():
    close, dividends, _ = _build_regime_series(n_months=50)
    inputs = {"FXA": {"unadjusted_close": close, "dividends": dividends}}
    with pytest.raises(ValueError, match="non-empty"):
        mod.make_signal_fn("FXA", close.index[-10:], {"FXA": pd.DataFrame()},
                           inputs=inputs)
    with pytest.raises(KeyError):
        mod.make_signal_fn("FXA", close.index[-10:], {}, inputs=inputs)


# ------------------------------------------------------------- fetcher

def _fetcher():
    import importlib.util
    path = os.path.join(_ROOT, "scripts",
                        "fetch_batch5_events_fx_carry_distribution_yield_regime.py")
    assert os.path.exists(path), path
    spec = importlib.util.spec_from_file_location("_fx_carry_fetcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)               # no network at import
    return module


def test_fetcher_validate_close_rejects_non_positive_or_non_finite():
    fetcher = _fetcher()
    idx = pd.bdate_range("2020-01-01", periods=5)
    ok = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
    fetcher.validate_close(ok)                    # no raise
    bad = ok.copy()
    bad.iloc[2] = 0.0
    with pytest.raises(ValueError, match="positive"):
        fetcher.validate_close(bad)
    bad2 = ok.copy()
    bad2.iloc[2] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        fetcher.validate_close(bad2)


def test_fetcher_validate_dividends_rejects_non_positive_and_duplicates():
    fetcher = _fetcher()
    idx = pd.DatetimeIndex(["2020-01-15", "2020-02-15"])
    ok = pd.Series([0.1, 0.2], index=idx)
    fetcher.validate_dividends(ok)                 # no raise
    zero = ok.copy()
    zero.iloc[0] = 0.0
    with pytest.raises(ValueError, match="positive"):
        fetcher.validate_dividends(zero)
    dup_idx = pd.DatetimeIndex(["2020-01-15", "2020-01-15"])
    dup = pd.Series([0.1, 0.2], index=dup_idx)
    with pytest.raises(ValueError, match="unique"):
        fetcher.validate_dividends(dup)


def test_fetcher_build_payload_and_provenance_round_trip():
    fetcher = _fetcher()
    idx = pd.bdate_range("2020-01-01", periods=10)
    close = pd.Series(np.linspace(10.0, 11.0, 10), index=idx)
    div_idx = pd.DatetimeIndex(["2020-01-02", "2020-01-08"])
    dividends = pd.Series([0.05, 0.06], index=div_idx)
    payload = fetcher.build_payload(close, dividends)
    assert list(payload) == ["FXA"]
    assert len(payload["FXA"]["unadjusted_close"]) == 10
    assert len(payload["FXA"]["dividends"]) == 2
    assert payload["FXA"]["dividends"]["2020-01-02"] == pytest.approx(0.05)
    provenance = fetcher.build_provenance(close, dividends, fetched_at="2026-09-15T00:00:00Z")
    assert provenance["ticker"] == "FXA"
    assert provenance["row_count_unadjusted_close"] == 10
    assert provenance["event_count_dividends"] == 2
    assert provenance["distribution_events_per_year"] == {"2020": 2}
    assert "$" not in json.dumps(provenance)
    assert "$" not in json.dumps(payload)
