"""Tests for src/mining/batch5/gold_copper_ratio_fear_valuation_spy_overlay.py.

Chunk layout: contract shell, the ex-ante calendar and the lagged futures read,
the rolling-z state (with the hole rules), the merged-window construction, the
frozen futures cache, then (further down) the curve builder and the two control
arms. The account-curve arithmetic itself is tested in test_mining_long_overlay.py;
the bespoke driver in test_run_gold_copper_overlay_abi1.py. This family is ALWAYS
run through its own driver (scripts/run_gold_copper_overlay_abi1.py), never the
generic scripts/run_abi1_mining.py dispatch.

Every fixture is SYNTHETIC and production-shaped (actual NYSE sessions); nothing
here reads the network or a real price. Anything that writes goes to tmp_path.
The family's catalog entry lives in data/mining/catalog.json.
"""
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC = os.path.join(_ROOT, "src")
_SCRIPTS = os.path.join(_ROOT, "scripts")
for _path in (_SRC, _SCRIPTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mining import long_overlay as lo  # noqa: E402
from mining.batch5 import gold_copper_ratio_fear_valuation_spy_overlay as mod  # noqa: E402
from mining.batch5 import treasury_month_end_index_demand_overlay as cal  # noqa: E402

_FAMILY = "gold_copper_ratio_fear_valuation_spy_overlay"
_CATALOG_ENTRY_PATH = os.path.join(_ROOT, "data", "mining", "catalog.json")  # the family's shipped catalog entry


def _catalog_entry_from(fh):
    """This family's entry in the shipped catalog."""
    return next(entry for entry in json.load(fh) if entry["family_id"] == _FAMILY)


_DOLLAR_FIGURE = re.compile(r"\$\s?\d")
_EIGHT_DIGITS = re.compile(r"(?<![\d.])\b\d{8}\b(?![\d.])")  # word-bounded, so a hex digest is not a hit
_REAL_FILES = [os.path.join(_ROOT, "data", "mining", "trial_ledger.jsonl")] + [
    os.path.join(_ROOT, "data", "mining", "gold_copper", name)
    for name in ("manifest.json", "yahoo_GC_F.csv", "yahoo_HG_F.csv")]


def _snapshot():
    return {p: ((os.stat(p).st_size, os.stat(p).st_mtime_ns) if os.path.exists(p) else None) for p in _REAL_FILES}


@pytest.fixture(scope="module", autouse=True)
def _the_real_ledger_and_the_real_frozen_cache_are_untouched_by_this_suite():
    """Every write in this file must land under tmp_path: the real trial ledger and
    the real frozen futures cache must be byte-for-byte (size and mtime) unchanged
    when the module finishes."""
    before = _snapshot()
    yield
    assert _snapshot() == before


def _catalog_entry() -> dict:
    with open(_CATALOG_ENTRY_PATH, "r", encoding="utf-8") as fh:
        return _catalog_entry_from(fh)


def _sessions(start, end):
    return cal.exchange_sessions(pd.Timestamp(start), pd.Timestamp(end))


def _close_series(index, seed, vol, start=100.0, drift=0.0):
    rs = np.random.RandomState(seed)
    r = rs.normal(drift, vol, len(index))
    r[0] = 0.0
    return pd.Series(start * np.cumprod(1.0 + r), index=index)


def _futures(start="2000-08-30", end="2009-12-31", seed=11):
    """A production-shaped gold and copper close pair on the REAL exchange
    sessions (both listed from ``start``)."""
    idx = _sessions(start, end)
    gc = _close_series(idx, seed, 0.008, start=300.0)
    hg = _close_series(idx, seed + 1, 0.012, start=90.0)
    return gc, hg


# -------------------------------------------------------------- contract

def test_module_constants_match_the_candidate_contract():
    assert mod.FAMILY_ID == _FAMILY
    assert mod.SHAPE == "single_long"               # contract LABEL only, see docstring
    assert mod.TRADED == ["SPY"]
    assert mod.SIGNAL == ["GC=F", "HG=F", "^IRX"]   # signal-only series, never traded
    assert mod.SNAPSHOTS == {}
    assert mod.BENCHMARK == "SPY"
    assert isinstance(mod.EXPECTED_MIN_TRADES, int) and mod.EXPECTED_MIN_TRADES > 0
    assert callable(mod.make_signal_fn)


def test_the_module_passes_the_batch5_contract_validator():
    import run_batch5_mining as batch5
    spec = batch5.validate_module_contract(mod)
    assert spec["family_id"] == _FAMILY and spec["shape"] == "single_long"


def test_make_signal_fn_always_raises_so_the_generic_dispatch_cannot_score_it():
    with pytest.raises(RuntimeError, match="run_gold_copper_overlay_abi1"):
        mod.make_signal_fn()
    with pytest.raises(RuntimeError):
        mod.make_signal_fn("SPY", {"SPY": None})


def test_parameters_are_tied_to_the_constants_they_describe():
    p = mod.PARAMETERS
    assert p["overlay_notional_fraction"] == mod.OVERLAY_NOTIONAL == 0.25
    assert p["z_window_month_ends"] == mod.Z_WINDOW_MONTHS == 60
    assert p["state_threshold_z"] == mod.Z_THRESHOLD == 0.0
    assert p["signal_lag_actual_sessions"] == mod.SIGNAL_LAG_SESSIONS == 1
    assert p["max_asof_staleness_calendar_days"] == mod.MAX_ASOF_STALENESS_DAYS == 4
    assert p["benchmark"] == "SPY" == mod.BENCHMARK
    assert mod.OVERLAY_NOTIONAL <= lo.MAX_OVERLAY_NOTIONAL
    assert p["financing_rate"].startswith("^IRX") and "no spread" in p["financing_rate"]
    assert "merged" in p["window"] and "fixed" in p["window"]      # the drift is stated in the parameters


def test_catalog_entry_is_complete_honest_and_new():
    from mining import catalog as catalog_mod
    entry = _catalog_entry()
    for field in catalog_mod.REQUIRED_FIELDS:
        assert field in entry, field
    assert entry["family_id"] == _FAMILY
    assert entry["status"] == "eligible"
    assert entry["account_types"] == ["taxable"]        # the overlay is margin-financed
    assert entry["tickers"] == ["SPY", "GC=F", "HG=F", "^IRX"]
    assert entry["timeframe"] == "daily"
    assert entry["added_date"] == "2026-09-24"
    source = entry["source"]
    assert "5677761" in source and "Roh" in source and "Datastream" in source
    assert "unverified" in source.lower()               # journal status of the working paper
    assert "Parnes" in source
    mechanism = entry["mechanism"].lower()
    for phrase in ("overlay", "spy", "one session", "merged", "fixed share", "control"):
        assert phrase in mechanism, phrase
    for neighbour in ("vix_spike_reversal_equity_buy", "real_rate_gold_equity_rotation",
                      "commodity_currency_leadlag"):
        assert neighbour in entry["distinctness_note"]
    assert not _DOLLAR_FIGURE.search(json.dumps(entry))     # no dollar figures
    assert not _EIGHT_DIGITS.search(json.dumps(entry))      # no YYYYMMDD integers


def test_the_catalog_text_records_every_binding_amendment_and_disclosure():
    """Every binding amendment and the honest limits must be readable in the
    permanent record."""
    text = json.dumps(_catalog_entry()).lower()
    for phrase in (
            "one window", "fixed share", "up to 57",                # merged windows
            "always-on", "random-window",                           # control arms
            "scheduled", "exchange_calendars", "raise",             # calendar and holes
            "one session", "settlement",                            # read lag
            "tariff", "no winsor",                                  # vendor splice
            "handful of high episodes", "datastream", "level predictor",   # thin evidence
            "single pre-declared", "60-month", "pipeline's own"):   # own constants
        assert phrase in text, phrase


def test_the_module_docstring_states_the_amendments_and_the_departures():
    doc = mod.__doc__
    for phrase in ("Amendments applied", "one-session lag", "settlement", "fixed share count",
                   "always-on", "random-window", "report-only", "Datastream", "chosen blind",
                   "risk-basis rule", "overlay-only excess account", "Sandy"):
        assert phrase in doc, phrase


def test_no_dollar_figure_or_yyyymmdd_literal_in_the_module_source():
    with open(mod.__file__, "r", encoding="utf-8") as fh:
        source = fh.read()
    assert not _DOLLAR_FIGURE.search(source)
    assert not _EIGHT_DIGITS.search(source)


# ---------------------------------------------------------- the calendar

def test_the_month_end_is_the_scheduled_last_session_never_read_off_the_data():
    ends = mod.month_end_schedule("2021-11-01", "2022-01-15")
    assert [d.strftime("%Y-%m-%d") for d in ends] == ["2021-11-30", "2021-12-31", "2022-01-31"]
    # 2022-01-01 fell on a Saturday and the NYSE did not close on Friday 12-31, so
    # the schedule keeps it; the calendar takes no data argument at all
    assert pd.Timestamp("2021-12-31") in ends


def test_the_schedule_keeps_the_unscheduled_sandy_closure_as_a_scheduled_session():
    """October 2012: the 29th and 30th were unscheduled closures (not known in
    advance), so the SCHEDULED month-end is still the 31st and the schedule is the
    treasury sibling's regular-holidays-only rule."""
    ends = mod.month_end_schedule("2012-10-01", "2012-10-31")
    assert list(ends) == [pd.Timestamp("2012-10-31")]


@pytest.mark.parametrize("month_end,expected", [
    ("2005-07-29", "2005-07-28"),      # an ordinary Friday: the Thursday before
    ("2018-12-31", "2018-12-28"),      # a Monday month-end: the previous Friday
    ("2012-10-31", "2012-10-26"),      # Sandy: the 29th and 30th were CLOSED, so the last real session
    ("2003-11-28", "2003-11-26"),      # Thanksgiving Thursday is closed
])
def test_the_read_date_is_the_last_actual_session_strictly_before_the_month_end(month_end, expected):
    got = mod.read_date_for(month_end)
    assert got == pd.Timestamp(expected)
    assert got < pd.Timestamp(month_end)


# ------------------------------------------------------ rolling z / state

def test_z_is_defined_only_from_the_sixtieth_read_and_matches_the_definition():
    rs = np.random.RandomState(4)
    x = pd.Series(rs.normal(5.0, 0.3, 75), index=pd.date_range("2000-08-31", periods=75, freq="ME"))
    z = mod.rolling_z(x)
    assert z.iloc[:59].isna().all() and z.iloc[59:].notna().all()
    window = x.iloc[0:60]
    assert z.iloc[59] == pytest.approx((x.iloc[59] - window.mean()) / window.std(ddof=1))
    window = x.iloc[10:70]
    assert z.iloc[69] == pytest.approx((x.iloc[69] - window.mean()) / window.std(ddof=1))
    assert mod.Z_WINDOW_MONTHS == 60


def test_the_window_includes_the_current_read_and_nothing_later():
    rs = np.random.RandomState(9)
    x = pd.Series(rs.normal(0, 1, 80), index=pd.date_range("2000-08-31", periods=80, freq="ME"))
    z_full = mod.rolling_z(x)
    for k in (59, 65, 79):
        assert mod.rolling_z(x.iloc[: k + 1]).iloc[k] == pytest.approx(z_full.iloc[k])
    # a change AFTER k cannot move z_k
    y = x.copy()
    y.iloc[70:] += 100.0
    assert mod.rolling_z(y).iloc[65] == pytest.approx(z_full.iloc[65])


def test_high_iff_the_read_is_at_or_above_the_mean_of_its_own_60_reads():
    x = pd.Series(np.zeros(80), index=pd.date_range("2000-08-31", periods=80, freq="ME"))
    x.iloc[:] = np.tile([0.0, 1.0], 40)                # mean 0.5 in every full window
    z = mod.rolling_z(x)
    state = mod.state_from_z(z)
    assert state.iloc[59:].tolist() == [bool(v >= 0) for v in z.iloc[59:]]
    assert bool(state.iloc[61]) is True and bool(state.iloc[60]) is False   # 1.0 above, 0.0 below the mean
    assert state.iloc[:59].isna().all()                # not defined during warm-up
    assert mod.Z_THRESHOLD == 0.0


def test_a_constant_window_has_no_defined_z_and_refuses_instead_of_guessing():
    x = pd.Series(np.full(70, 5.0), index=pd.date_range("2000-08-31", periods=70, freq="ME"))
    with pytest.raises(ValueError, match="zero variance"):
        mod.rolling_z(x)


# ------------------------------------------------- the lagged futures read

def _signal(gc, hg, last="2009-12-31"):
    return mod.monthly_signal(gc, hg, last_session=pd.Timestamp(last))


def test_the_first_decision_is_the_sixtieth_monthly_read_and_reads_start_in_the_first_data_month():
    gc, hg = _futures()
    table = _signal(gc, hg)
    first = table.index[table["defined"]][0]
    assert first == pd.Timestamp("2005-07-29")             # Aug 2000 .. Jul 2005 is 60 reads
    assert table.index[0] == pd.Timestamp("2000-08-31")
    assert table.loc[pd.Timestamp("2005-07-29"), "read_date"] == pd.Timestamp("2005-07-28")
    assert not table.loc[:pd.Timestamp("2005-06-30"), "defined"].any()
    assert table["defined"].iloc[-1] and table.index[-1] == pd.Timestamp("2009-12-31")


def test_the_read_never_touches_the_bar_of_the_month_end_itself_or_later():
    """The one-session lag: perturbing EVERY futures bar on or after each
    scheduled month-end leaves every x, z and state unchanged, while perturbing
    the read-date bar moves that month's x."""
    gc, hg = _futures()
    base = _signal(gc, hg)
    # perturb everything on/after ONE month-end at a time (a later month's own read
    # is legitimately moved by its own perturbation, so the check is per month)
    for month_end in list(base.index[62:70]):
        gc3, hg3 = gc.copy(), hg.copy()
        gc3.loc[gc3.index >= month_end] *= 3.0
        hg3.loc[hg3.index >= month_end] *= 0.4
        got = _signal(gc3, hg3)
        upto = got.loc[:month_end]
        pd.testing.assert_series_equal(upto["x"].iloc[-1:], base.loc[:month_end]["x"].iloc[-1:])
        assert upto["state"].iloc[-1] == base.loc[:month_end]["state"].iloc[-1]
    month_end = base.index[70]
    gc4 = gc.copy()
    gc4.loc[base.loc[month_end, "read_date"]] *= 2.0
    assert _signal(gc4, hg).loc[month_end, "x"] != base.loc[month_end, "x"]


def test_x_is_the_log_gold_copper_ratio_of_the_read_date_closes():
    gc, hg = _futures()
    table = _signal(gc, hg)
    month_end = pd.Timestamp("2004-03-31")
    read = table.loc[month_end, "read_date"]
    assert table.loc[month_end, "x"] == pytest.approx(np.log(gc.loc[read] / hg.loc[read]))


def test_a_decision_needs_the_month_end_session_to_be_at_or_before_the_last_session():
    gc, hg = _futures(end="2010-03-31")
    table = _signal(gc, hg, last="2010-03-30")           # the March month-end (the 31st) is not reached
    assert table.index[-1] == pd.Timestamp("2010-02-26")
    table = _signal(gc, hg, last="2010-03-31")
    assert table.index[-1] == pd.Timestamp("2010-03-31")


# ------------------------------------------------------------- hole rules

def _drop(series, dates):
    return series.drop(index=pd.DatetimeIndex(dates))


def test_a_warm_up_hole_uses_a_causal_as_of_read_and_is_reported():
    gc, hg = _futures()
    table0 = _signal(gc, hg)
    read = table0.loc[pd.Timestamp("2003-05-30"), "read_date"]
    gc_holed = _drop(gc, [read])
    table = _signal(gc_holed, hg)
    row = table.loc[pd.Timestamp("2003-05-30")]
    assert row["gc_row_date"] < read and (read - row["gc_row_date"]).days <= mod.MAX_ASOF_STALENESS_DAYS
    prior = gc_holed.loc[:read].iloc[-1]
    assert row["x"] == pytest.approx(np.log(prior / hg.loc[read]))     # the earlier close, never a later one
    assert any(r["series"] == "GC=F" and r["read_date"] == read.strftime("%Y-%m-%d") for r in table.attrs["asof_reads"])


def test_an_as_of_read_staler_than_the_limit_is_refused_in_warm_up_too():
    gc, hg = _futures()
    read = _signal(gc, hg).loc[pd.Timestamp("2003-05-30"), "read_date"]
    dates = [d for d in gc.index if read - pd.Timedelta(days=6) <= d <= read]
    with pytest.raises(mod.FuturesReadError) as info:
        _signal(_drop(gc, dates), hg)
    assert info.value.reason_code == "futures_asof_too_stale"
    assert read.strftime("%Y-%m-%d") in str(info.value)


def test_a_hole_after_the_first_defined_z_is_refused_never_patched():
    gc, hg = _futures()
    table0 = _signal(gc, hg)
    month_end = pd.Timestamp("2007-10-31")
    read = table0.loc[month_end, "read_date"]
    for holed in ((_drop(gc, [read]), hg), (gc, _drop(hg, [read]))):
        with pytest.raises(mod.FuturesReadError) as info:
            _signal(*holed)
        assert info.value.reason_code == "futures_read_hole"
        assert read.strftime("%Y-%m-%d") in info.value.missing_dates


def test_the_hole_rule_starts_exactly_at_the_first_defined_z():
    """The read of the first decision month (July 2005) is the first one that may
    not be as-of; the read of the month before it still may be."""
    gc, hg = _futures()
    table0 = _signal(gc, hg)
    just_before = table0.loc[pd.Timestamp("2005-06-30"), "read_date"]
    _signal(_drop(gc, [just_before]), hg)                           # tolerated (warm-up)
    first = table0.loc[pd.Timestamp("2005-07-29"), "read_date"]
    with pytest.raises(mod.FuturesReadError):
        _signal(_drop(gc, [first]), hg)                             # refused (first defined z)


def test_futures_history_ending_before_the_last_decision_read_is_refused():
    gc, hg = _futures(end="2009-11-30")
    with pytest.raises(mod.FuturesReadError) as info:
        _signal(gc, hg, last="2009-12-31")
    assert info.value.reason_code == "futures_read_hole"


def test_the_futures_series_are_validated():
    gc, hg = _futures()
    bad = gc.copy()
    bad.iloc[100] = -1.0
    with pytest.raises(mod.FuturesReadError) as info:
        _signal(bad, hg)
    assert info.value.reason_code == "futures_series_invalid"
    dup = pd.concat([gc, gc.iloc[-1:]])
    with pytest.raises(mod.FuturesReadError):
        _signal(dup, hg)


# ------------------------------------------------------ merged windows

def _month_ends(n):
    ends = pd.date_range("2005-07-29", periods=n, freq="ME")
    return pd.DatetimeIndex(ends)


def test_consecutive_high_months_merge_into_one_window_ending_at_the_first_non_high_month_end():
    ends = _month_ends(9)
    states = [False, True, True, False, True, False, True, True]
    windows = mod.high_windows(ends, states)
    assert windows == [(ends[1], ends[3]), (ends[4], ends[5]), (ends[6], ends[8])]   # the last runs to the terminal exit


def test_windows_never_touch_or_overlap_so_the_engine_cannot_raise():
    rs = np.random.RandomState(2)
    for _ in range(50):
        states = list(rs.rand(40) > 0.4)
        ends = _month_ends(41)
        windows = mod.high_windows(ends, states)
        for (a, b), (c, d) in zip(windows, windows[1:]):
            assert a < b < c < d
        assert all(entry < exit_ for entry, exit_ in windows)


def test_all_high_is_one_window_and_all_low_is_none():
    ends = _month_ends(6)
    assert mod.high_windows(ends, [True] * 5) == [(ends[0], ends[5])]
    assert mod.high_windows(ends, [False] * 5) == []
    with pytest.raises(ValueError):
        mod.high_windows(ends, [True] * 4)          # month_ends must be one longer than states


def test_windows_report_the_merged_episode_lengths_in_months():
    states = [False, True, True, False, True, False, True, True]
    assert mod.episode_lengths(states) == [2, 1, 2]
    assert mod.episode_lengths([False] * 3) == []
    assert mod.episode_lengths([True] * 4) == [4]


# --------------------------------------------------- the frozen futures cache

def _fake_fetcher(frames):
    def fetch(symbol):
        frame = frames[symbol].copy()
        frame.index = pd.DatetimeIndex(frame.index).tz_localize("America/New_York")     # like yfinance
        return frame
    return fetch


def _ohlcv(close):
    return pd.DataFrame({"Open": close.shift(1).fillna(close.iloc[0]), "High": close * 1.002,
                         "Low": close * 0.998, "Close": close, "Volume": 100}, index=close.index)


def test_freeze_writes_only_under_the_given_directory_with_a_sha256_manifest(tmp_path):
    gc, hg = _futures(end="2001-12-31")
    frames = {"GC=F": _ohlcv(gc), "HG=F": _ohlcv(hg)}
    manifest = mod.freeze_futures(str(tmp_path / "frozen"), fetcher=_fake_fetcher(frames),
                                  through="2001-12-31", retrieved_utc="2026-09-24T12:00:00Z")
    files = sorted(os.listdir(tmp_path / "frozen"))
    assert files == ["manifest.json", "yahoo_GC_F.csv", "yahoo_HG_F.csv"]
    for name, entry in manifest["files"].items():
        raw = (tmp_path / "frozen" / name).read_bytes()
        assert entry["sha256"] == hashlib.sha256(raw).hexdigest() and entry["bytes"] == len(raw)
        assert entry["rows"] == len(frames[entry["symbol"]]) and entry["last_date"] == "2001-12-31"
    on_disk = json.loads((tmp_path / "frozen" / "manifest.json").read_text())
    assert on_disk == manifest and manifest["through"] == "2001-12-31"


def test_freeze_drops_bars_after_the_through_date_the_live_partial_bar(tmp_path):
    gc, hg = _futures(end="2001-12-31")
    frames = {"GC=F": _ohlcv(gc), "HG=F": _ohlcv(hg)}
    mod.freeze_futures(str(tmp_path), fetcher=_fake_fetcher(frames), through="2001-12-20")
    loaded_gc, loaded_hg, manifest = mod.load_frozen_futures(str(tmp_path))
    assert loaded_gc.index[-1] == pd.Timestamp("2001-12-20") and loaded_hg.index[-1] == pd.Timestamp("2001-12-20")
    assert manifest["files"]["yahoo_GC_F.csv"]["last_date"] == "2001-12-20"


def test_load_round_trips_the_closes_exactly_and_verifies_the_hash(tmp_path):
    gc, hg = _futures(end="2001-12-31")
    frames = {"GC=F": _ohlcv(gc), "HG=F": _ohlcv(hg)}
    mod.freeze_futures(str(tmp_path), fetcher=_fake_fetcher(frames), through="2001-12-31")
    loaded_gc, loaded_hg, _ = mod.load_frozen_futures(str(tmp_path))
    pd.testing.assert_series_equal(loaded_gc, gc, check_names=False, check_freq=False)
    pd.testing.assert_series_equal(loaded_hg, hg, check_names=False, check_freq=False)
    assert loaded_gc.index.tz is None


def test_a_tampered_cache_file_is_refused_with_a_reason_code(tmp_path):
    gc, hg = _futures(end="2001-12-31")
    mod.freeze_futures(str(tmp_path), fetcher=_fake_fetcher({"GC=F": _ohlcv(gc), "HG=F": _ohlcv(hg)}),
                       through="2001-12-31")
    path = tmp_path / "yahoo_GC_F.csv"
    path.write_bytes(path.read_bytes().replace(b"2001-12-31", b"2001-12-30", 1))
    with pytest.raises(mod.FuturesReadError) as info:
        mod.load_frozen_futures(str(tmp_path))
    assert info.value.reason_code == "futures_cache_hash_mismatch"


def test_a_missing_cache_or_manifest_is_refused_with_a_reason_code(tmp_path):
    with pytest.raises(mod.FuturesReadError) as info:
        mod.load_frozen_futures(str(tmp_path))
    assert info.value.reason_code == "futures_cache_missing"
    gc, hg = _futures(end="2001-12-31")
    mod.freeze_futures(str(tmp_path / "f"), fetcher=_fake_fetcher({"GC=F": _ohlcv(gc), "HG=F": _ohlcv(hg)}),
                       through="2001-12-31")
    os.remove(tmp_path / "f" / "yahoo_HG_F.csv")
    with pytest.raises(mod.FuturesReadError) as info:
        mod.load_frozen_futures(str(tmp_path / "f"))
    assert info.value.reason_code == "futures_cache_missing"


def test_freeze_never_overwrites_a_frozen_splice_unless_told_to(tmp_path):
    gc, hg = _futures(end="2001-12-31")
    frames = {"GC=F": _ohlcv(gc), "HG=F": _ohlcv(hg)}
    mod.freeze_futures(str(tmp_path), fetcher=_fake_fetcher(frames), through="2001-12-31")
    before = (tmp_path / "yahoo_GC_F.csv").read_bytes()
    with pytest.raises(FileExistsError):
        mod.freeze_futures(str(tmp_path), fetcher=_fake_fetcher(frames), through="2001-12-31")
    assert (tmp_path / "yahoo_GC_F.csv").read_bytes() == before
    mod.freeze_futures(str(tmp_path), fetcher=_fake_fetcher(frames), through="2001-12-31", overwrite=True)


def test_freeze_refuses_a_vendor_payload_it_cannot_use_and_writes_nothing(tmp_path):
    gc, hg = _futures(end="2001-12-31")
    bad = _ohlcv(hg)
    bad.iloc[5, bad.columns.get_loc("Close")] = np.nan
    with pytest.raises(mod.FuturesReadError) as info:
        mod.freeze_futures(str(tmp_path / "x"), fetcher=_fake_fetcher({"GC=F": _ohlcv(gc), "HG=F": bad}),
                           through="2001-12-31")
    assert info.value.reason_code == "futures_series_invalid"
    assert not (tmp_path / "x").exists() or os.listdir(tmp_path / "x") == []


# =======================================================================
# Chunk B: the account curve and the two control arms
# =======================================================================

def _level_path(n_months):
    """A piecewise ln(gold/copper) level per month: base, a high plateau, a dip, a
    second high plateau, a low finish -- so the state has merged multi-month
    HIGH runs, isolated switches and a LOW tail."""
    level = np.full(n_months, 5.0)
    level[65:76] = 5.6
    level[76:86] = 4.9
    level[86:96] = 5.6
    level[96:] = 4.8
    return level


def _world(end="2011-06-30", seed=21, spy_start="2002-01-02"):
    """Production-shaped futures, SPY and ^IRX on the real exchange sessions. The
    log gold/copper ratio of every day is its calendar month's level plus a small
    noise, so a month-end read carries that month's level."""
    idx = _sessions("2000-08-30", end)
    rs = np.random.RandomState(seed)
    hg = _close_series(idx, seed, 0.010, start=100.0)
    month_number = (idx.year - 2000) * 12 + idx.month - 8
    level = _level_path(int(month_number.max()) + 1)[month_number]
    gc = hg * np.exp(level + rs.normal(0.0, 0.01, len(idx)))
    spy_idx = _sessions(spy_start, end)
    spy_close = _close_series(spy_idx, seed + 2, 0.010, start=100.0, drift=0.0004).to_numpy()
    open_ = np.concatenate([[spy_close[0]], spy_close[:-1]])
    spy = pd.DataFrame({"Open": open_, "High": np.maximum(open_, spy_close) * 1.001,
                        "Low": np.minimum(open_, spy_close) * 0.999, "Close": spy_close}, index=spy_idx)
    irx_idx = _sessions("2000-01-03", end)
    irx_close = np.linspace(1.0, 4.0, len(irx_idx))
    irx = pd.DataFrame({"Open": irx_close, "High": irx_close, "Low": irx_close, "Close": irx_close}, index=irx_idx)
    return gc, hg, spy, irx


_COSTS = dict(spy_cost_bps=2.0, spy_entry_cost_bps=1.0)


def _built(end="2011-06-30", **world_kwargs):
    gc, hg, spy, irx = _world(end=end, **world_kwargs)
    curve = mod.build_curve(spy, irx, gc, hg, **_COSTS)
    ctx = mod.prepare(spy, irx, gc, hg)
    return gc, hg, spy, irx, curve, ctx


def test_prepare_lays_the_account_on_spys_sessions_from_the_first_defined_z_month_end():
    gc, hg, spy, irx, curve, ctx = _built()
    assert ctx["first_decision"] == pd.Timestamp("2005-07-29")
    assert ctx["idx"][0] == pd.Timestamp("2005-07-29") and ctx["idx"][-1] == spy.index[-1]
    assert curve["strategy_daily"].index.equals(ctx["idx"])
    assert len(ctx["states"]) == len(ctx["month_ends"]) - 1 > 60
    assert ctx["month_ends"][-1] > spy.index[-1]                       # the terminal exit lies after the data
    assert ctx["month_ends"][-1] == mod.month_end_schedule(spy.index[-1] + pd.Timedelta(days=1),
                                                            spy.index[-1] + pd.Timedelta(days=5))[0]


def test_the_trades_are_exactly_the_merged_high_windows():
    _, _, spy, _, curve, ctx = _built()
    trades = curve["trades_df"]
    windows = mod.high_windows(ctx["month_ends"], ctx["states"])
    assert len(windows) >= 3
    assert max(mod.episode_lengths(ctx["states"])) >= 5              # a genuinely merged multi-month run
    assert len(trades) == len(windows) - int(curve["window_info"]["entered_on_last_bar"])
    for (entry, exit_), (_, trade) in zip(windows, trades.iterrows()):
        assert trade["entry_date"] == entry                            # entry at the close of the first HIGH month-end
        if exit_ <= spy.index[-1]:
            assert trade["exit_date"] == exit_ and trade["reason"] == "window"
        else:
            assert trade["exit_date"] == spy.index[-1] and trade["reason"] == "end_of_data"
    assert curve["account_kind"] == "benchmark_plus_overlay"          # overlay-only account basis


def test_the_curve_carries_the_window_info_and_the_signal_table():
    _, _, spy, _, curve, ctx = _built()
    info = curve["window_info"]
    assert info["windows"] == mod.high_windows(ctx["month_ends"], ctx["states"])
    assert info["n_decisions"] == len(ctx["states"]) and info["n_high"] == int(np.sum(ctx["states"]))
    assert info["episode_months"] == mod.episode_lengths(ctx["states"])
    assert info["truncated_final"] in (True, False) and info["entered_on_last_bar"] in (True, False)
    assert isinstance(curve["signal_table"], pd.DataFrame)
    assert info["first_decision"] == "2005-07-29"


def test_the_account_equals_the_evaluators_benchmark_series_outside_the_windows():
    """SPY lists long before the account starts (production shape). Outside the
    windows the account must equal, bit for bit, the benchmark series the ABI-1
    evaluator subtracts, computed on SPY's FULL history."""
    from mining.benchmark_relative import cost_adjusted_benchmark_daily_returns
    _, _, spy, _, curve, _ = _built()
    bench = cost_adjusted_benchmark_daily_returns(spy, entry_cost_bps=_COSTS["spy_entry_cost_bps"])
    account = curve["strategy_daily"]
    outside = ~curve["overlay_mask"].to_numpy()
    assert spy.index[0] < account.index[0]
    assert outside.sum() > 100 and (~outside).sum() > 100
    assert np.array_equal(account.to_numpy()[outside], bench.reindex(account.index).to_numpy()[outside])


def test_the_overlay_is_a_quarter_of_entry_equity_in_spy_at_fixed_shares_financed_at_the_bill_rate():
    _, _, spy, irx, curve, ctx = _built()
    trades = curve["trades_df"]
    idx = ctx["idx"]
    entry = trades["entry_date"].iloc[0]
    pos = idx.get_loc(entry)
    account = curve["strategy_daily"].to_numpy()
    equity = np.cumprod(1.0 + account)                                 # equity path from the account's first close
    e_entry = equity[pos]
    close = spy["Close"].loc[idx].to_numpy()
    shares = mod.OVERLAY_NOTIONAL * e_entry / close[pos]
    fin = lo.causal_financing_rate(irx["Close"], idx).to_numpy()
    elapsed = np.concatenate([[1.0], np.diff(idx.values.astype("datetime64[D]")).astype(float)])
    for t in (pos + 1, pos + 2, pos + 7):
        expected = (shares * (close[t] - close[t - 1])
                    - mod.OVERLAY_NOTIONAL * e_entry * fin[t] * elapsed[t] / 365.0) / equity[t - 1]
        assert curve["overlay_daily"].iloc[t] == pytest.approx(expected, rel=1e-9, abs=1e-12)
    assert not curve["overlay_mask"].iloc[pos]                         # nothing is earned on the entry session itself


def _spy_on_a_path(spy, growth_per_day):
    path = 100.0 * np.exp(growth_per_day * np.arange(len(spy)))
    out = spy.copy()
    out["Close"] = path
    out["Open"] = np.concatenate([[path[0]], path[:-1]])
    out["High"] = np.maximum(out["Open"], path) * 1.0001
    out["Low"] = np.minimum(out["Open"], path) * 0.9999
    return out


@pytest.mark.parametrize("growth,expect_below", [(0.0006, True), (-0.0006, False)])
def test_a_fixed_share_count_lets_the_overlay_share_of_equity_drift_as_the_closed_form_says(growth, expect_below):
    """The disclosed drift (amendment 1), proven rather than asserted. Fixed
    overlay shares o = 0.25 c and a fixed borrowed principal, with the overlay's
    daily P&L reinvested in the SPY core (the builder's own account identity
    E_t = E_{t-1}(1 + r_t) + o dP_t), give EXACTLY (financing and costs switched
    off) the overlay's share of account equity
        s_t = 0.25 / (1 + 0.25 * sum_k (1 - P_{k-1} / P_k)),
    about 0.25 / (1 + 0.25 ln x) for SPY at x times its entry price: it falls
    slowly as SPY rises and rises as SPY falls."""
    gc, hg, spy, irx = _world(end="2011-06-30")
    irx = irx.assign(Close=0.0, Open=0.0, High=0.0, Low=0.0)
    path = _spy_on_a_path(spy, growth)
    ctx = mod.prepare(path, irx, gc, hg)
    curve = mod.curve_from_states(ctx, path, [True] * len(ctx["states"]), **_COSTS)
    idx = ctx["idx"]
    equity = np.cumprod(1.0 + curve["strategy_daily"].to_numpy())
    close = path["Close"].loc[idx].to_numpy()
    shares = mod.OVERLAY_NOTIONAL * equity[0] / close[0]
    share_of_equity = shares * close / equity
    steps = np.concatenate([[0.0], 1.0 - close[:-1] / close[1:]])[1:]
    exact = mod.OVERLAY_NOTIONAL / (1.0 + mod.OVERLAY_NOTIONAL * np.concatenate([[0.0], np.cumsum(steps)]))
    assert share_of_equity[0] == pytest.approx(mod.OVERLAY_NOTIONAL)
    assert np.allclose(share_of_equity[:-1], exact[:-1], rtol=1e-9)    # the last session also pays the exit cost
    x = close[-1] / close[0]
    assert share_of_equity[-2] == pytest.approx(mod.OVERLAY_NOTIONAL / (1.0 + mod.OVERLAY_NOTIONAL * np.log(x)), rel=2e-3)
    if expect_below:
        assert share_of_equity[-1] < mod.OVERLAY_NOTIONAL
    else:
        assert share_of_equity[-1] > 0.30
    assert len(curve["trades_df"]) == 1                                # one merged window held throughout


def test_all_low_states_give_an_account_that_is_exactly_spy():
    from mining.benchmark_relative import cost_adjusted_benchmark_daily_returns
    gc, hg, spy, irx, _, ctx = _built()
    curve = mod.curve_from_states(ctx, spy, [False] * len(ctx["states"]), **_COSTS)
    bench = cost_adjusted_benchmark_daily_returns(spy, entry_cost_bps=_COSTS["spy_entry_cost_bps"])
    assert np.array_equal(curve["strategy_daily"].to_numpy(), bench.reindex(ctx["idx"]).to_numpy())
    assert len(curve["trades_df"]) == 0 and not curve["overlay_mask"].any()


def test_curve_from_states_refuses_a_state_vector_of_the_wrong_length():
    gc, hg, spy, irx, _, ctx = _built()
    with pytest.raises(ValueError, match="one state per decision"):
        mod.curve_from_states(ctx, spy, [True] * (len(ctx["states"]) - 1), **_COSTS)


def test_the_overlay_cost_is_the_spy_leg_cost_and_scales_with_it():
    gc, hg, spy, irx = _world()
    a = mod.build_curve(spy, irx, gc, hg, spy_cost_bps=2.0, spy_entry_cost_bps=1.0)["trades_df"]
    b = mod.build_curve(spy, irx, gc, hg, spy_cost_bps=6.0, spy_entry_cost_bps=1.0)["trades_df"]
    # cost_ret is already per unit of the sizing equity: notional x round-trip cost
    assert b["cost_ret"].iloc[0] == pytest.approx(3.0 * a["cost_ret"].iloc[0], rel=1e-9)
    assert a["cost_ret"].iloc[0] == pytest.approx(mod.OVERLAY_NOTIONAL * 2.0 / 1e4, rel=1e-9)


# ---------------------------- data-calendar refusals and the final window

def test_a_spy_history_that_starts_after_the_first_decision_is_refused_with_a_reason_code():
    gc, hg, spy, irx = _world(spy_start="2006-01-03")
    with pytest.raises(lo.CalendarCoverageError) as info:
        mod.prepare(spy, irx, gc, hg)
    assert info.value.reason_code == "spy_history_starts_after_first_decision"


def test_a_missing_exchange_session_inside_the_account_span_is_refused_not_patched():
    gc, hg, spy, irx = _world()
    hole = spy.index[1200]
    with pytest.raises(lo.CalendarCoverageError) as info:
        mod.prepare(spy.drop(index=hole), irx, gc, hg)
    assert info.value.reason_code == "missing_exchange_session"
    assert hole.strftime("%Y-%m-%d") in info.value.missing_dates


def test_a_stray_bar_on_a_closed_day_is_refused():
    gc, hg, spy, irx = _world()
    stray = pd.Timestamp("2008-12-25")
    extra = spy.iloc[[spy.index.get_loc(spy.index[spy.index > "2008-12-20"][0])]].copy()
    extra.index = pd.DatetimeIndex([stray])
    with pytest.raises(lo.CalendarCoverageError) as info:
        mod.prepare(pd.concat([spy, extra]).sort_index(), irx, gc, hg)
    assert info.value.reason_code == "bar_on_closed_day"


def test_a_missing_decision_month_end_session_is_refused_and_never_shifted():
    gc, hg, spy, irx = _world()
    ctx = mod.prepare(spy, irx, gc, hg)
    month_end = ctx["month_ends"][10]
    with pytest.raises(lo.CalendarCoverageError) as info:
        mod.prepare(spy.drop(index=month_end), irx, gc, hg)
    assert month_end.strftime("%Y-%m-%d") in info.value.missing_dates


def test_a_futures_hole_propagates_from_the_signal_as_a_reason_coded_refusal():
    gc, hg, spy, irx = _world()
    read = mod.monthly_signal(gc, hg, last_session=spy.index[-1]).loc[pd.Timestamp("2007-10-31"), "read_date"]
    with pytest.raises(mod.FuturesReadError) as info:
        mod.prepare(spy, irx, gc.drop(index=read), hg)
    assert info.value.reason_code == "futures_read_hole"


def _last_high_start(ctx):
    """Index k of the LAST month whose state is HIGH while the previous one is LOW."""
    states = [bool(s) for s in ctx["states"]]
    return max(k for k in range(1, len(states)) if states[k] and not states[k - 1])


def test_data_ending_on_a_window_entry_date_opens_nothing_and_is_reported():
    gc, hg, spy, irx = _world(end="2011-06-30")
    ctx = mod.prepare(spy, irx, gc, hg)
    k = _last_high_start(ctx)
    cut = ctx["decisions"][k]
    gc2, hg2, spy2, irx2 = gc.loc[:cut], hg.loc[:cut], spy.loc[:cut], irx.loc[:cut]
    curve = mod.build_curve(spy2, irx2, gc2, hg2, **_COSTS)
    info = curve["window_info"]
    assert info["entered_on_last_bar"] is True and info["truncated_final"] is False
    assert len(curve["trades_df"]) == len(info["windows"]) - 1
    assert info["windows"][-1][0] == cut


def test_data_ending_inside_an_open_window_is_truncated_final_and_closed_at_the_last_close():
    gc, hg, spy, irx = _world(end="2011-06-30")
    ctx = mod.prepare(spy, irx, gc, hg)
    k = _last_high_start(ctx)
    cut = spy.index[spy.index.get_loc(ctx["decisions"][k]) + 3]
    gc2, hg2, spy2, irx2 = gc.loc[:cut], hg.loc[:cut], spy.loc[:cut], irx.loc[:cut]
    curve = mod.build_curve(spy2, irx2, gc2, hg2, **_COSTS)
    info = curve["window_info"]
    assert info["truncated_final"] is True and info["entered_on_last_bar"] is False
    last = curve["trades_df"].iloc[-1]
    assert last["reason"] == "end_of_data" and last["exit_date"] == cut


def test_a_decision_month_end_after_the_last_bar_is_not_a_decision():
    gc, hg, spy, irx = _world(end="2011-06-30")
    cut = pd.Timestamp("2011-06-29")                                    # the day before the June month-end
    ctx = mod.prepare(spy.loc[:cut], irx.loc[:cut], gc.loc[:cut], hg.loc[:cut])
    assert ctx["decisions"][-1] == pd.Timestamp("2011-05-31")


# --------------------------------------------------------- always-on control

def test_the_always_on_control_is_the_same_construction_with_the_state_forced_high():
    gc, hg, spy, irx, curve, ctx = _built()
    control = mod.build_always_on_curve(ctx, spy, **_COSTS)
    assert len(control["trades_df"]) == 1
    trade = control["trades_df"].iloc[0]
    assert trade["entry_date"] == ctx["first_decision"] and trade["exit_date"] == spy.index[-1]
    assert trade["reason"] == "end_of_data"
    assert control["overlay_mask"].iloc[1:].all() and not control["overlay_mask"].iloc[0]
    assert control["account_kind"] == "benchmark_plus_overlay"
    same = mod.curve_from_states(ctx, spy, [True] * len(ctx["states"]), **_COSTS)
    assert np.array_equal(control["strategy_daily"].to_numpy(), same["strategy_daily"].to_numpy())
    # same financing series, same cost model as the main arm
    assert control["financing_rate"].equals(curve["financing_rate"])
    assert control["trades_df"]["cost_ret"].iloc[0] == pytest.approx(
        mod.OVERLAY_NOTIONAL * _COSTS["spy_cost_bps"] / 1e4, rel=1e-9)


# ------------------------------------------------------ random-window control

def test_the_random_window_draws_are_deterministic_and_seed_dependent():
    states = [False, True, True, False, True, False, False, True, True, True, False, False, True]
    a = mod.random_window_state_draws(states, 25, seed=7)
    b = mod.random_window_state_draws(states, 25, seed=7)
    c = mod.random_window_state_draws(states, 25, seed=8)
    assert a == b and a != c and len(a) == 25


def test_every_draw_keeps_the_episode_count_the_exact_length_multiset_and_the_high_share():
    rs = np.random.RandomState(3)
    for trial in range(30):
        states = list(rs.rand(60) > 0.55)
        for draw in mod.random_window_state_draws(states, 20, seed=trial + 1):
            assert len(draw) == len(states)
            assert sorted(mod.episode_lengths(draw)) == sorted(mod.episode_lengths(states))
            assert sum(draw) == sum(states)


def test_draw_episodes_are_never_adjacent_so_windows_never_touch():
    states = [True, True, False, True, False, True, True, True, False, False]
    for draw in mod.random_window_state_draws(states, 200, seed=5):
        ends = _month_ends(len(draw) + 1)
        windows = mod.high_windows(ends, draw)
        assert len(windows) == len(mod.episode_lengths(states))
        for (a, b), (c, d) in zip(windows, windows[1:]):
            assert a < b < c < d


def test_the_draws_place_episodes_at_varied_positions_and_cover_the_edges():
    states = [False] * 10 + [True, True] + [False] * 8
    draws = mod.random_window_state_draws(states, 400, seed=1)
    starts = {d.index(True) for d in draws}
    assert len(starts) >= 12 and 0 in starts and 18 in starts          # the whole feasible range, both edges
    assert len({tuple(d) for d in draws}) >= 12


def test_the_random_draws_do_not_touch_numpys_global_random_state():
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    mod.random_window_state_draws([True, False, True], 10, seed=4)
    assert np.array_equal(np.random.get_state()[1], before)


def test_no_episode_means_no_window_in_any_draw_and_a_full_high_run_stays_one_window():
    assert mod.random_window_state_draws([False] * 5, 3, seed=1) == [[False] * 5] * 3
    assert all(all(d) for d in mod.random_window_state_draws([True] * 5, 3, seed=1))


def test_the_default_control_configuration_is_pre_declared_and_tied_to_the_parameters():
    assert mod.RANDOM_CONTROL_DRAWS >= 200
    assert mod.PARAMETERS["random_window_control_draws"] == mod.RANDOM_CONTROL_DRAWS
    assert int(mod.PARAMETERS["random_window_control_seed"].replace("_", "")) == mod.RANDOM_CONTROL_SEED
    assert mod.RANDOM_CONTROL_PERCENTILE_BAR == 0.95
    assert "500" in mod.PARAMETERS["control_arms"]


def test_the_random_control_curve_uses_the_same_builder_and_reset_structure_as_the_main_arm():
    gc, hg, spy, irx, curve, ctx = _built()
    draw = mod.random_window_state_draws(ctx["states"], 3, seed=mod.RANDOM_CONTROL_SEED)[0]
    control = mod.curve_from_states(ctx, spy, draw, **_COSTS)
    assert control["account_kind"] == "benchmark_plus_overlay"
    assert len(control["window_info"]["windows"]) == len(curve["window_info"]["windows"])
    assert control["overlay_mask"].sum() != 0


def test_build_curve_accepts_a_prepared_context_and_gives_the_identical_curve():
    gc, hg, spy, irx, curve, ctx = _built()
    again = mod.build_curve(spy, irx, gc, hg, context=ctx, **_COSTS)
    assert np.array_equal(again["strategy_daily"].to_numpy(), curve["strategy_daily"].to_numpy())
    assert again["window_info"]["windows"] == curve["window_info"]["windows"]


@pytest.mark.parametrize("last,expected_terminal", [
    ("2005-12-30", "2006-01-31"),      # the last decision month-end is a Friday, the 31st a Saturday
    ("2006-09-29", "2006-10-31"),      # likewise (30 September 2006 was a Saturday)
    ("2011-06-30", "2011-07-29"),      # the last decision IS the last calendar day
])
def test_the_terminal_exit_is_the_next_months_scheduled_month_end_whatever_day_the_last_one_fell_on(
        last, expected_terminal):
    """Guards the terminal-exit lookup on odd cut dates: looking it up from the day
    after the last decision would stay in the SAME calendar month whenever the last
    scheduled session is not the last calendar day."""
    gc, hg, spy, irx = _world(end="2011-06-30")
    cut = pd.Timestamp(last)
    ctx = mod.prepare(spy.loc[:cut], irx.loc[:cut], gc.loc[:cut], hg.loc[:cut])
    assert ctx["decisions"][-1] == cut
    assert ctx["terminal_exit"] == pd.Timestamp(expected_terminal) and ctx["terminal_exit"] > cut
    curve = mod.curve_from_states(ctx, spy.loc[:cut], [True] * len(ctx["states"]), **_COSTS)     # must not raise
    assert curve["window_info"]["windows"][-1][1] == pd.Timestamp(expected_terminal)


def test_fewer_than_sixty_monthly_reads_is_a_reason_coded_refusal_not_a_bare_error():
    gc, hg, spy, irx = _world(end="2004-06-30")           # 47 monthly reads: z is never defined
    with pytest.raises(mod.FuturesReadError) as info:
        mod.prepare(spy, irx, gc, hg)
    assert info.value.reason_code == "futures_history_too_short"
    assert "60" in str(info.value)


def test_a_read_exactly_at_the_mean_is_high_the_threshold_is_at_or_above_zero():
    state = mod.state_from_z(pd.Series([-1e-9, 0.0, 1e-9, np.nan]))
    assert [bool(v) for v in state.iloc[:3]] == [False, True, True] and pd.isna(state.iloc[3])


def test_a_spy_history_missing_the_first_decision_session_itself_is_refused_not_shifted():
    """SPY lacking exactly the FIRST decision month-end leaves the span (which then
    starts the next session) with no interior hole, so only the explicit
    decision-session check can catch it; the first window must never slide to the
    next available bar."""
    gc, hg, spy, irx = _world()
    first = mod.prepare(spy, irx, gc, hg)["first_decision"]
    with pytest.raises(lo.CalendarCoverageError) as info:
        mod.prepare(spy.drop(index=first), irx, gc, hg)
    assert info.value.reason_code == "missing_month_end_session"
    assert first.strftime("%Y-%m-%d") in info.value.missing_dates


def test_the_draws_permute_the_episode_order_not_only_the_positions():
    states = [True, False, False, True, True, True, True, True, False, False, False, False]      # lengths 1 and 5
    orders = {tuple(mod.episode_lengths(d)) for d in mod.random_window_state_draws(states, 200, seed=3)}
    assert orders == {(1, 5), (5, 1)}


# =======================================================================
# Additional checks: seam alignment, the staleness edge,
# account prefix-invariance, the calendar provenance and the disclosed figures
# =======================================================================

def _by_hand_states(gc, hg, last_decision):
    """The signal recomputed from the raw frames with NOTHING from the module under
    test but the shared calendar (the sibling module's scheduled and actual session
    sets): per scheduled month-end M, x = ln(GC/HG) on the last actual session strictly
    before M, z against the last 60 reads incl. this one (sample stdev), HIGH iff
    z >= 0. Returns ``{month_end: bool}`` for every month from the 60th read on."""
    first = pd.Timestamp("2000-08-01")
    scheduled = cal.scheduled_sessions(first, last_decision)
    actual = cal.exchange_sessions(first - pd.Timedelta(days=30), last_decision)
    per_month = scheduled.to_period("M")
    ends = [scheduled[per_month == p][-1] for p in per_month.unique()]
    xs = []
    for month_end in ends:
        read = actual[actual < month_end][-1]
        xs.append(float(np.log(gc.loc[read] / hg.loc[read])))
    out = {}
    for k in range(59, len(ends)):
        window = np.array(xs[k - 59:k + 1])
        out[ends[k]] = bool((xs[k] - window.mean()) / window.std(ddof=1) >= 0.0)
    return out


def _by_hand_windows(states_by_month, decisions, terminal):
    """(entry, exit) per run of consecutive HIGH decision month-ends, written out
    directly: entry at the first HIGH month-end of a run, exit at the first month-end
    that is not HIGH (or the terminal exit)."""
    windows, start = [], None
    for month_end in decisions:
        high = states_by_month[month_end]
        if high and start is None:
            start = month_end
        elif not high and start is not None:
            windows.append((start, month_end))
            start = None
    if start is not None:
        windows.append((start, terminal))
    return windows


def test_the_state_at_each_decision_month_end_is_the_state_of_the_window_that_enters_there():
    """A one-month state delay in ``prepare`` would be invisible to a test that
    compares the trades only with ``high_windows`` of the SAME possibly shifted
    list, so the alignment is pinned end to end against a
    recomputation from the raw frames, then against hand-checkable facts of the
    synthetic level path: the second plateau (ln ratio 5.6 for October 2007 to July
    2008, 4.9 in September 2007 and 4.8 from August 2008) enters at the October
    2007 month-end and exits at the August 2008 month-end."""
    gc, hg, spy, irx, curve, ctx = _built()
    by_hand = _by_hand_states(gc, hg, ctx["decisions"][-1])
    table = ctx["table"]
    assert list(ctx["decisions"]) == [d for d in by_hand]                     # the same decision months
    for i, month_end in enumerate(ctx["decisions"]):
        assert bool(table.loc[month_end, "state"]) is by_hand[month_end], month_end
        assert ctx["states"][i] is by_hand[month_end], month_end              # prepare's vector, month by month
    windows = _by_hand_windows(by_hand, ctx["decisions"], ctx["terminal_exit"])
    trades = curve["trades_df"]
    assert len(trades) == len(windows) >= 3
    for (entry, exit_), (_, trade) in zip(windows, trades.iterrows()):
        assert trade["entry_date"] == entry and trade["exit_date"] == exit_
    # hand-checkable plateau facts (independent of the noise the synthetic world carries)
    assert (pd.Timestamp("2007-10-31"), pd.Timestamp("2008-08-29")) in windows
    assert by_hand[pd.Timestamp("2007-09-28")] is False and by_hand[pd.Timestamp("2007-10-31")] is True
    assert by_hand[pd.Timestamp("2008-07-31")] is True and by_hand[pd.Timestamp("2008-08-29")] is False
    assert any(exit_ == pd.Timestamp("2006-12-29") for _, exit_ in windows)   # the first plateau ends with December 2006
    assert pd.Timestamp("2007-10-31") in set(trades["entry_date"])


@pytest.mark.parametrize("stale_days,tolerated", [(1, True), (4, True), (5, False), (6, False)])
def test_the_as_of_staleness_limit_is_pinned_at_its_edge(stale_days, tolerated):
    """an as-of read exactly ``MAX_ASOF_STALENESS_DAYS`` (4) days
    old is tolerated and one exactly 5 days old raises, so neither ``>=`` nor
    ``> MAX + 1`` survives. The April 2003 read date is a Tuesday (29th); the last
    gold row left is 28th (1 day), 25th (4), 24th (5) or 23rd (6 days back)."""
    gc, hg = _futures()
    read = _signal(gc, hg).loc[pd.Timestamp("2003-04-30"), "read_date"]
    assert read == pd.Timestamp("2003-04-29") and read.weekday() == 1
    kept = read - pd.Timedelta(days=stale_days)
    assert kept in gc.index                                            # a real session is the last row left
    holed = _drop(gc, [d for d in gc.index if kept < d <= read])
    assert holed.loc[:read].index[-1] == kept
    if tolerated:
        table = _signal(holed, hg)
        reads = [r for r in table.attrs["asof_reads"] if r["series"] == "GC=F" and r["read_date"] == "2003-04-29"]
        assert [r["staleness_days"] for r in reads] == [stale_days]
        assert table.loc[pd.Timestamp("2003-04-30"), "gc_row_date"] == kept
    else:
        with pytest.raises(mod.FuturesReadError) as info:
            _signal(holed, hg)
        assert info.value.reason_code == "futures_asof_too_stale"


def _world_frames_after(world, cutoff, seed):
    """The world with EVERYTHING strictly after ``cutoff`` perturbed: SPY closes get
    a large random walk of their own, the bill rate jumps by five points and both
    futures are rescaled at random. Nothing on or before ``cutoff`` changes."""
    gc, hg, spy, irx = world
    rs = np.random.RandomState(seed)
    gc2, hg2 = gc.copy(), hg.copy()
    later = gc2.index > cutoff
    gc2[later] = gc2[later] * np.exp(rs.normal(0.0, 0.3, int(later.sum())))
    hg2[later] = hg2[later] * np.exp(rs.normal(0.0, 0.3, int(later.sum())))
    spy2 = spy.copy()
    close = spy2["Close"].to_numpy().copy()
    after = np.flatnonzero(spy2.index > cutoff)
    for pos in after:
        close[pos] = close[pos - 1] * float(np.exp(rs.normal(0.0, 0.05)))
    spy2["Close"] = close
    spy2["Open"] = np.concatenate([[close[0]], close[:-1]])
    spy2["High"] = np.maximum(spy2["Open"], close) * 1.001
    spy2["Low"] = np.minimum(spy2["Open"], close) * 0.999
    irx2 = irx.copy()
    irx2.loc[irx2.index > cutoff, ["Open", "High", "Low", "Close"]] += 5.0
    return gc2, hg2, spy2, irx2


def test_the_account_through_any_cutoff_does_not_depend_on_anything_after_it():
    """end-to-end prefix invariance of the ACCOUNT curve. For each
    cutoff T (1 and 2 sessions after a window entry, and the sessions around
    window-exit and ordinary month-ends, so a peek of only one or two sessions
    would cross T), (a) CUTTING SPY, the bill series and both futures after T and
    (b) PERTURBING all four after T leave every account session through T unchanged
    (in the cut, the last session is excluded: an open window closes there and pays
    its exit cost) and the windows entered up to T identical."""
    world = _world(end="2011-06-30")
    gc, hg, spy, irx = world
    base_ctx = mod.prepare(spy, irx, gc, hg)
    base = mod.build_curve(spy, irx, gc, hg, context=base_ctx, **_COSTS)
    sessions = spy.index
    windows = base["window_info"]["windows"]
    assert len(windows) >= 3
    at = lambda d, k: sessions[sessions.get_loc(d) + k]                     # noqa: E731
    cutoffs = set()
    for entry, exit_ in windows[:3]:
        cutoffs |= {at(entry, 1), at(entry, 2), at(exit_, -1), exit_, at(exit_, 1)}
    for month_end in (base_ctx["decisions"][12], base_ctx["decisions"][40]):        # ordinary month-ends
        cutoffs |= {at(month_end, k) for k in (-2, -1, 0, 1, 2)}
    cutoffs = sorted(c for c in cutoffs if c > base_ctx["first_decision"] + pd.Timedelta(days=5)
                     and c < sessions[-1] - pd.Timedelta(days=5))
    assert len(cutoffs) >= 20
    base_daily = base["strategy_daily"]
    for n, cutoff in enumerate(cutoffs):
        # (a) cut every input at the cutoff
        cut = mod.build_curve(spy.loc[:cutoff], irx.loc[:cutoff], gc.loc[:cutoff], hg.loc[:cutoff], **_COSTS)
        daily = cut["strategy_daily"]
        assert daily.index[-1] == cutoff
        assert np.array_equal(daily.iloc[:-1].to_numpy(), base_daily.loc[daily.index[:-1]].to_numpy()), cutoff
        assert [w[0] for w in cut["window_info"]["windows"]] == [w[0] for w in windows if w[0] <= cutoff], cutoff
        # (b) keep the shape but perturb everything strictly after the cutoff
        p_gc, p_hg, p_spy, p_irx = _world_frames_after(world, cutoff, seed=n)
        pert = mod.build_curve(p_spy, p_irx, p_gc, p_hg, **_COSTS)
        pdaily = pert["strategy_daily"]
        assert np.array_equal(pdaily.loc[:cutoff].to_numpy(), base_daily.loc[:cutoff].to_numpy()), cutoff
        assert [w[0] for w in pert["window_info"]["windows"] if w[0] <= cutoff] == \
            [w[0] for w in windows if w[0] <= cutoff], cutoff
    # the perturbation is real: it moves the account after the last cutoff
    assert not np.array_equal(pdaily.to_numpy(), base_daily.to_numpy())


def _reference_schedule_text(decisions, reads, terminal):
    lines = [f"{m:%Y-%m-%d}|{r:%Y-%m-%d}" for m, r in zip(decisions, reads)]
    lines.append(f"terminal|{terminal:%Y-%m-%d}")
    return "\n".join(lines) + "\n"


def test_the_schedule_provenance_records_the_calendar_version_and_a_hash_of_the_dates():
    """every decision month-end, its read date and the terminal exit
    are functions of the installed calendar package, so the ledger note records the
    package version and a sha256 of that exact text."""
    import importlib.metadata
    gc, hg, spy, irx, curve, ctx = _built()
    prov = mod.schedule_provenance(ctx)
    reads = [ctx["table"].loc[m, "read_date"] for m in ctx["decisions"]]
    text = _reference_schedule_text(ctx["decisions"], reads, ctx["terminal_exit"])
    assert prov["schedule_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert prov["exchange_calendars_version"] == importlib.metadata.version("exchange_calendars")
    assert prov["n_decisions"] == len(ctx["decisions"]) and prov["terminal_exit"] == ctx["terminal_exit"].strftime("%Y-%m-%d")
    assert re.fullmatch(r"[0-9a-f]{64}", prov["schedule_sha256"])
    # moving ONE read date by one session moves the hash
    table = ctx["table"].copy()
    table.loc[ctx["decisions"][10], "read_date"] = table.loc[ctx["decisions"][10], "read_date"] - pd.Timedelta(days=1)
    assert mod.schedule_provenance({**ctx, "table": table})["schedule_sha256"] != prov["schedule_sha256"]
    # moving the terminal exit moves it too
    assert mod.schedule_provenance({**ctx, "terminal_exit": ctx["terminal_exit"] + pd.Timedelta(days=1)}
                                   )["schedule_sha256"] != prov["schedule_sha256"]


# split at a digit run so no long run of digits appears in the source
_SCHEDULE_2005_2026_SHA256 = ("bde9364683f7e2f349" "07486a2ec3b867ebb55444e4a0d07ab18b207aa722790b")


def test_the_reference_calendar_schedule_is_pinned_as_a_whole_set():
    """a calendar-package upgrade that edited historical exchange
    holiday data would silently move month-ends and read dates. The whole
    calendar-only schedule of a fixed reference calendar (254 decision
    month-ends from July 2005 through August 2026, their read dates and the terminal
    September 2026 exit) is pinned by its sha256; it is also recomputed here straight
    from the exchange calendar (last real session of each month; the read date is the
    session before it), so the pin and the module cannot drift together."""
    import exchange_calendars as xc
    assert len(_SCHEDULE_2005_2026_SHA256) == 64
    ends = mod.month_end_schedule("2005-07-01", "2026-09-30")
    assert len(ends) == 255 and ends[0] == pd.Timestamp("2005-07-29") and ends[-1] == pd.Timestamp("2026-09-30")
    reads = mod.read_dates_for(ends)
    digest = mod.schedule_digest(ends[:-1], reads[:-1], ends[-1])
    assert digest == _SCHEDULE_2005_2026_SHA256
    xnys = xc.get_calendar("XNYS", start=pd.Timestamp("2005-06-01"), end=pd.Timestamp("2026-09-30"))
    sessions = pd.DatetimeIndex(xnys.sessions)
    last_of_month = pd.Series(sessions, index=sessions).groupby(sessions.to_period("M")).last()
    last_of_month = last_of_month[last_of_month.index >= pd.Period("2005-07", "M")]
    independent_reads = [sessions[sessions.get_loc(d) - 1] for d in last_of_month]
    text = _reference_schedule_text(list(last_of_month.iloc[:-1]), independent_reads[:-1], last_of_month.iloc[-1])
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == _SCHEDULE_2005_2026_SHA256


def _flat(text: str) -> str:
    return " ".join(text.split())


def _catalog_text() -> str:
    """The catalog entry's prose (its source, mechanism and note fields), whitespace-flattened."""
    entry = _catalog_entry()
    return _flat(" ".join(str(entry[k]) for k in ("source", "mechanism", "distinctness_note")))


_NO_TAX_PHRASE = "pays no modelled tax (long-term at the 0.0 rate), not merely deferred"


def test_the_catalog_and_docstring_list_the_untaxed_long_term_windows_among_the_flattering_assumptions():
    flat_rate = "windows held over 365 days are untaxed at the pipeline's 0.0 long-term rate"
    assert flat_rate in _catalog_text() and flat_rate in _flat(mod.__doc__)


def test_the_interpretation_rule_is_stated_unchanged_but_tax_asymmetric_in_the_docstring_and_the_catalog():
    for name, text in (("catalog", _catalog_text()), ("docstring", _flat(mod.__doc__))):
        assert "tax-asymmetric" in text, name
        assert "the pre-declared interpretation rule itself is unchanged" in text, name
    assert "exceeds the always-on arm's AND sits at or above the 95th percentile" in _flat(mod.__doc__)


def test_the_docstring_states_the_measured_zero_volume_read_dates():
    """Measured read-only on the frozen splice: 9 gold and 1 copper read dates fall on a
    zero-volume row, 11 and 2 counting the as-of warm-up reads that land on one."""
    text = _flat(mod.__doc__)
    assert "9 gold and 1 copper read dates (11 and 2 counting the as-of warm-up reads)" in text


def test_the_yahoo_close_is_described_as_consistent_with_the_settlement_time():
    """The evidence supports only 'consistent with' (gold tight, copper noisier)."""
    text = _catalog_text()
    assert "consistent with the COMEX settlement time" in text
    assert "gold tight, copper noisier" in text


def test_the_always_on_sharpe_is_attributed_to_an_unconditional_estimate_before_the_pipeline_frictions():
    """any always-on Sharpe figure quoted is an unconditional estimate, before this
    pipeline's lag, financing, costs and tax, not this pipeline's own measurement."""
    doc = _flat(mod.__doc__)
    assert "excess Sharpe near" not in doc          # no measured Sharpe figure in the module
    assert "unconditional baseline" in doc
    text = _catalog_text()
    if "excess Sharpe near" in text:                # any figure the catalog quotes stays attributed
        assert "unconditional estimate" in text and "before this pipeline's lag, financing, costs and tax" in text


def _section(markdown: str, heading_start: str) -> str:
    parts = re.split(r"(?m)^## ", markdown)
    hits = [p for p in parts if p.startswith(heading_start)]
    assert len(hits) == 1, (heading_start, len(hits))
    return hits[0]


def test_the_distinctness_note_claims_only_that_no_gold_copper_ratio_is_in_the_catalog():
    """A sibling family uses copper open interest, so the note claims only that
    no gold/copper ratio appears anywhere in the catalog."""
    text = _catalog_text()
    assert "no gold/copper ratio anywhere in the catalog" in text
    assert "a sibling family uses copper open interest" in text


def test_a_zero_variance_window_is_a_reason_coded_futures_read_refusal_not_a_bare_value_error():
    """A ValueError subclass that is also reason-coded, so a driver reports a clean
    refusal instead of a traceback."""
    x = pd.Series(np.full(70, 5.0), index=pd.date_range("2000-08-31", periods=70, freq="ME"))
    with pytest.raises(mod.FuturesReadError, match="zero variance") as caught:
        mod.rolling_z(x)
    assert caught.value.reason_code == "futures_window_zero_variance"
    assert isinstance(caught.value, ValueError)
