"""Tests for src/mining/batch5/ex_dividend_day_capture_pooled.py.

Synthetic frames and synthetic event tables only; no network, and no real
events file is read (the loader is monkeypatched where the factory would call
it).
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
from mining.batch5 import ex_dividend_day_capture_pooled as mod  # noqa: E402

MODULE_PATH = os.path.join(_ROOT, "src", "mining", "batch5",
                           "ex_dividend_day_capture_pooled.py")
TICKER = "SYN"


# ------------------------------------------------------------ synthetic data

def _frame(n: int = 600, start: str = "2018-01-02", seed: int = 0) -> pd.DataFrame:
    """Business-day OHLC, prices near 100 with a small random walk."""
    index = pd.bdate_range(start, periods=n)
    rng = np.random.RandomState(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.005, n)))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.001, n))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close},
                        index=index)


def _record(index: pd.DatetimeIndex, pos: int, yield_value: float, *,
            close: float = 100.0, signal_pos=None) -> dict:
    """An events.json record for an ex-date at frame position ``pos`` whose
    amount / unadjusted_close_signal equals ``yield_value``."""
    signal_pos = pos - 2 if signal_pos is None else signal_pos
    return {"ex_date": index[pos].strftime("%Y-%m-%d"),
            "amount": yield_value * close,
            "signal_date": (index[signal_pos].strftime("%Y-%m-%d")
                            if signal_pos is not None and signal_pos >= 0 else None),
            "unadjusted_close_signal": close,
            "yield": yield_value}


# positions of the synthetic ex-dates and what each is meant to prove
ABOVE_1, ABOVE_2, BELOW, AT_FLOOR, ABOVE_3 = 50, 120, 200, 300, 420


def _events(index: pd.DatetimeIndex) -> dict:
    records = [
        _record(index, ABOVE_1, 0.010),          # fires
        _record(index, ABOVE_2, 0.0062),         # fires
        _record(index, BELOW, 0.0049),           # below the floor: skipped
        _record(index, AT_FLOOR, 0.005),         # exactly at the floor: fires (>=)
        _record(index, ABOVE_3, 0.020),          # fires
        _record(index, 1, 0.030, signal_pos=None),  # fewer than 2 prior sessions
        {"ex_date": "2018-01-06", "amount": 1.0, "signal_date": "2018-01-04",
         "unadjusted_close_signal": 100.0, "yield": 0.01},  # a Saturday: not a session
        {"ex_date": index[500].strftime("%Y-%m-%d"), "amount": 1.0,
         "signal_date": None, "unadjusted_close_signal": None,
         "yield": None},                          # no yield inputs: skipped
        {"ex_date": index[550].strftime("%Y-%m-%d"), "amount": 1.0,
         "signal_date": index[547].strftime("%Y-%m-%d"),
         "unadjusted_close_signal": 100.0, "yield": 0.01},  # calendar mismatch
    ]
    return {TICKER: records}


FIRING = (ABOVE_1, ABOVE_2, AT_FLOOR, ABOVE_3)


def _factory(frame: pd.DataFrame, index=None, events=None, ticker=TICKER):
    return mod.make_signal_fn(
        ticker, frame.index if index is None else index, {ticker: frame},
        events=_events(frame.index) if events is None else events)


def _signals(signal_fn, frame: pd.DataFrame, upto: int) -> list:
    return [(bool(s.entry), bool(s.exit)) for s in
            (signal_fn(frame.iloc[:i + 1], i) for i in range(upto + 1))]


# ------------------------------------------------------------------ contract

def test_driver_validates_module_contract():
    import run_batch5_mining as batch5
    module = batch5._import_module_file("ex_dividend_day_capture_pooled",
                                        MODULE_PATH)
    spec = batch5.validate_module_contract(module, repo_root=_ROOT)
    assert spec["family_id"] == mod.FAMILY_ID
    assert spec["shape"] == "event_pooled"
    assert spec["traded"] == mod.TRADED
    assert batch5.parameter_drift(module.PARAMETERS, mod.PARAMETERS) is None


# ------------------------------------------------------ rule: fires / flat

def test_event_signal_dates_select_e_minus_2_and_e_for_qualifying_events():
    frame = _frame()
    entries, exits, diag = mod.event_signal_dates(
        frame.index, _events(frame.index)[TICKER], mod.YIELD_FLOOR)
    assert entries == {frame.index[p - 2] for p in FIRING}
    assert exits == {frame.index[p] for p in FIRING}
    assert diag["qualifying"] == len(FIRING)
    assert diag["below_floor"] == 1
    assert diag["ex_date_not_in_calendar"] == 1
    assert diag["fewer_than_two_prior_sessions"] == 1
    assert diag["missing_yield_input"] == 1
    assert diag["signal_date_mismatch"] == 1
    assert diag["events_total"] == 9


def test_signal_fires_on_signal_bar_exits_on_ex_date_and_is_flat_elsewhere():
    frame = _frame()
    signal_fn = _factory(frame)
    got = _signals(signal_fn, frame, len(frame) - 1)
    for pos, (entry, exit_) in enumerate(got):
        assert entry == (pos + 2 in FIRING), pos
        assert exit_ == (pos in FIRING), pos
    # the sub-floor event and the malformed ones never produce a signal
    for pos in (BELOW, 1, 500, 550):
        assert got[pos] == (False, False)
        assert got[pos - 2] == (False, False)


def test_yield_is_recomputed_from_amount_and_unadjusted_close_not_trusted():
    frame = _frame()
    # A record whose stored yield claims to clear the floor but whose own
    # inputs do not must NOT fire: the rule is amount / close, not a label.
    forged = dict(_record(frame.index, ABOVE_1, 0.004))
    forged["yield"] = 0.5
    entries, _, diag = mod.event_signal_dates(frame.index, [forged], mod.YIELD_FLOOR)
    assert entries == set()
    assert diag["below_floor"] == 1


def test_ticker_without_events_is_always_flat():
    frame = _frame()
    signal_fn = _factory(frame, events={TICKER: []})
    assert all(s == (False, False) for s in _signals(signal_fn, frame, len(frame) - 1))
    signal_fn = _factory(frame, events={})  # ticker absent from the table
    assert all(s == (False, False) for s in _signals(signal_fn, frame, len(frame) - 1))


# ------------------------------------------------------------- look-ahead

def test_look_ahead_probe_future_bars_never_change_past_signals():
    frame = _frame()
    k = ABOVE_2 + 5
    before = _signals(_factory(frame), frame, k)
    perturbed = frame.copy()
    tail = perturbed.index[k + 1:]
    perturbed.loc[tail, "Close"] *= 1.3
    perturbed.loc[tail, "High"] *= 1.5
    perturbed.loc[tail, "Low"] *= 0.7
    perturbed.loc[tail, "Open"] *= 1.2
    after = _signals(_factory(perturbed), perturbed, k)
    assert before == after


def test_holdout_bucket_is_blind_to_bars_after_its_last_bar():
    frame = _frame()
    bucket = frame.index[250:520]
    events = _events(frame.index)  # the same event table for both builds
    full = _factory(frame, index=bucket, events=events)
    truncated_frame = frame.loc[frame.index <= bucket[-1]]
    truncated = _factory(truncated_frame, index=bucket, events=events)
    sub = frame.loc[bucket]
    assert _signals(full, sub, len(sub) - 1) == _signals(truncated, sub, len(sub) - 1)
    # and the bucket really sees the AT_FLOOR / ABOVE_3 events inside it
    got = _signals(full, sub, len(sub) - 1)
    for pos in (AT_FLOOR, ABOVE_3):
        rel = pos - 250
        assert got[rel - 2] == (True, False)
        assert got[rel] == (False, True)


def test_bucket_calendar_mismatch_is_refused():
    frame = _frame()
    signal_fn = _factory(frame)
    shifted = frame.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    with pytest.raises(ValueError):
        signal_fn(shifted.iloc[:10], 9)
    with pytest.raises(ValueError):
        signal_fn(frame.iloc[:10], len(frame) + 3)


# --------------------------------------------------------------- end to end

def test_end_to_end_run_backtest_one_bar_trades_on_each_qualifying_event():
    frame = _frame()
    trades, in_market = backtest_engine.run_backtest(
        frame, _factory(frame), cost_bps=2.0)
    assert len(trades) == len(FIRING)
    for pos, (_, row) in zip(FIRING, trades.iterrows()):
        assert row["entry_date"] == frame.index[pos - 1]    # cum-day open
        assert row["exit_date"] == frame.index[pos]         # ex-day close
        assert row["held"] == 1
        assert row["reason"] == "signal_exit"
        assert np.isclose(row["entry_px"], frame["Open"].iloc[pos - 1])
        assert np.isclose(row["exit_px"], frame["Close"].iloc[pos])
    assert int(in_market.sum()) == 2 * len(FIRING)


def test_end_to_end_pooled_two_tickers_run_independently():
    frame_a, frame_b = _frame(seed=1), _frame(seed=2)
    events = {"AAA": _events(frame_a.index)["SYN"],
              "BBB": [_record(frame_b.index, 77, 0.01)]}
    frames = {"AAA": frame_a, "BBB": frame_b}
    trades_a, _ = backtest_engine.run_backtest(
        frame_a, mod.make_signal_fn("AAA", frame_a.index, frames, events=events))
    trades_b, _ = backtest_engine.run_backtest(
        frame_b, mod.make_signal_fn("BBB", frame_b.index, frames, events=events))
    assert len(trades_a) == len(FIRING)
    assert len(trades_b) == 1
    assert trades_b["exit_date"].iloc[0] == frame_b.index[77]


# ----------------------------------------------------------------- data file
# The real events.json is not part of this public tree (it is regenerable from
# yfinance via scripts/fetch_batch5_events_ex_dividend_day_capture_pooled.py),
# so a fresh checkout lacks it.

def test_factory_default_uses_the_events_file(monkeypatch):
    frame = _frame()
    calls = []

    def fake_load():
        calls.append(1)
        return _events(frame.index)

    monkeypatch.setattr(mod, "load_events", fake_load)
    signal_fn = mod.make_signal_fn(TICKER, frame.index, {TICKER: frame})
    assert calls == [1]
    assert _signals(signal_fn, frame, ABOVE_1)[ABOVE_1 - 2] == (True, False)


# ------------------------------------------------- data hygiene: fail-closed
# One test per data-quality hazard the event table must refuse or neutralize.

def _payload(events=None, **overrides) -> dict:
    """A minimal well-formed events.json payload (every TRADED member keyed)."""
    payload = {
        "family_id": mod.FAMILY_ID,
        "schema_version": mod.EXPECTED_SCHEMA_VERSION,
        "basis_check": {"status": "verified", "violations": [],
                        "splits_checked": 10,
                        "directions": {name: {"detects": "..."}
                                       for name in mod.REQUIRED_BASIS_DIRECTIONS}},
        "events": {ticker: [] for ticker in mod.TRADED},
    }
    if events:
        payload["events"].update(events)
    payload.update(overrides)
    return payload


def test_load_events_accepts_a_well_formed_payload():
    assert mod.validate_events_payload(_payload()) is not None


def test_load_events_refuses_a_stale_schema_version():
    """A table on an older schema version carries no corporate-action
    evidence and no proven split basis. Refuse it, do not silently tolerate it."""
    with pytest.raises(ValueError, match="schema_version"):
        mod.validate_events_payload(_payload(schema_version=1))


@pytest.mark.parametrize("basis, expected", [
    (None, "basis_check"),
    ({"violations": []}, "basis_check did not pass"),
    ({"status": "verified",
      "violations": [{"ticker": "APH", "split_session": "2026-09-03"}]},
     "basis_check did not pass"),
])
def test_load_events_refuses_a_missing_or_failed_basis_check(basis, expected):
    """Hazard (APH, the 2026-09-03 2:1 split): Yahoo's dividend series had the
    split applied while its price history did not, so a table built from it
    stores every APH close on twice the amount basis and understates every
    APH yield by exactly 2x -- silently. A table that cannot prove the two
    bases agree is refused."""
    payload = _payload()
    if basis is None:
        del payload["basis_check"]
    else:
        payload["basis_check"] = basis
    with pytest.raises(ValueError, match=expected):
        mod.validate_events_payload(payload)


@pytest.mark.parametrize("directions", [
    None,
    {},
    {"price_history_unapplied": {"detects": "..."}},          # schema-2 shape
    {"dividend_series_unadjusted": {"detects": "..."}},
    {"price_history_unapplied": {}, "dividend_series_unadjusted": "not a dict"},
])
def test_load_events_refuses_a_one_directional_basis_check(directions):
    """A basis check that reads the PRICE side only (the schema-2 shape)
    can be recorded as a passing 'basis_check' for a table whose prices are
    already split-adjusted while its amounts are still pre-split -- which
    OVERSTATES every pre-split yield by the split ratio. A file that cannot
    name BOTH directions is refused."""
    payload = _payload()
    if directions is None:
        payload["basis_check"].pop("directions")
    else:
        payload["basis_check"]["directions"] = directions
    with pytest.raises(ValueError, match="directions|never read"):
        mod.validate_events_payload(payload)


def test_load_events_refuses_a_table_missing_a_pool_member():
    """A fetch can lose pool members to transient failures. An absent member
    produces no trades, so a partial table would quietly shrink the pooled
    sample instead of failing."""
    payload = _payload()
    del payload["events"]["AAPL"]
    with pytest.raises(ValueError, match="absent from the table"):
        mod.validate_events_payload(payload)


# ------------------------------------------------------ durable event_counts
# The per-member exclusion counts (below_floor, excluded_split_type_adjustment,
# signal_date_mismatch, events_dropped_secondary_share_class, ...) must be
# auditable from a run report, because the module's disclosure rests on them.
# The driver never reads signal_fn.event_counts, so the counts also land in a
# module-level sink that outlives the function (read via pooled_diagnostics).

def test_event_details_outlive_the_signal_function_and_pool_across_members():
    frame = _frame()
    records = _events(frame.index)[TICKER]
    events = {"AAA": records, "BBB": records, "GOOG": records,
              "GOOGL": records}
    frames = {name: frame for name in events}
    mod.reset_diagnostics()
    assert mod.pooled_diagnostics()["members"] == 0
    for ticker in events:
        signal_fn = mod.make_signal_fn(ticker, frame.index, frames, events=events)
        # the per-function attribute is populated as well
        assert signal_fn.event_counts["events_total"] in (0, len(records))
    del signal_fn

    by_ticker = mod.event_counts_by_ticker()
    assert sorted(by_ticker) == ["AAA", "BBB", "GOOG", "GOOGL"]
    pooled = mod.pooled_diagnostics()
    assert pooled["members"] == 4
    assert pooled["members_excluded_secondary_share_class"] == ["GOOG"]
    assert pooled["members_with_qualifying_events"] == 3
    totals = pooled["totals"]
    # three members carry the events; GOOG carries none but records what it
    # dropped, so the exclusion is visible instead of invisible
    assert totals["events_total"] == 3 * len(records)
    assert totals["qualifying"] == 3 * len(FIRING)
    assert totals["below_floor"] == 3
    assert totals["events_dropped_secondary_share_class"] == len(records)
    assert set(totals) == set(mod.EVENT_COUNTERS)
    assert json.dumps(pooled)  # a driver can serialise it into its report


def test_event_counters_partition_every_event_exactly():
    """The docstring's event_counts table is only auditable if the counters
    ACCOUNT for every record: one classification per event, none dropped
    silently. Asserted as an invariant so a future branch that forgets to
    count shows up as a partition that no longer closes."""
    frame = _frame()
    records = list(_events(frame.index)[TICKER])
    records.append(dict(_record(frame.index, ABOVE_2 + 30, 0.02),
                        split_on_ex_date=1.319))
    _, _, diag = mod.event_signal_dates(frame.index, records, mod.YIELD_FLOOR)
    classified = sum(diag[key] for key in (
        "qualifying", "below_floor", "ex_date_not_in_calendar",
        "fewer_than_two_prior_sessions", "missing_yield_input",
        "signal_date_mismatch", "excluded_split_type_adjustment"))
    assert diag["events_total"] == len(records) == classified


def test_rebuilding_a_member_for_a_second_window_does_not_double_count():
    """The driver builds a fresh signal function per window (in-sample and
    holdout) from the SAME full-history frame, so the sink must be keyed by
    ticker rather than appended to."""
    frame = _frame()
    mod.reset_diagnostics()
    for index in (frame.index, frame.index[250:520]):
        mod.make_signal_fn(TICKER, index, {TICKER: frame},
                           events=_events(frame.index))
    pooled = mod.pooled_diagnostics()
    assert pooled["members"] == 1
    assert pooled["totals"]["events_total"] == 9
    assert pooled["totals"]["qualifying"] == len(FIRING)


def test_split_type_adjustment_event_is_excluded_as_a_non_cash_distribution():
    """Hazard (DHR 2016-07-05, the Fortive spin-off): Yahoo carries it as a
    dividend of 24.56 AND a same-session split of 1.319, so auto-adjustment
    applies two factors to one corporate event and the one-bar "return" is
    +62.78%. Such an event must contribute no trade."""
    frame = _frame()
    cash = _record(frame.index, ABOVE_1, 0.010)
    spun = dict(cash, split_on_ex_date=1.319)

    entries, exits, diag = mod.event_signal_dates(
        frame.index, [spun], mod.YIELD_FLOOR)
    assert entries == set() and exits == set()
    assert diag["excluded_split_type_adjustment"] == 1
    assert diag["qualifying"] == 0

    entries, exits, diag = mod.event_signal_dates(
        frame.index, [cash], mod.YIELD_FLOOR)
    assert entries == {frame.index[ABOVE_1 - 2]}
    assert exits == {frame.index[ABOVE_1]}
    assert diag["excluded_split_type_adjustment"] == 0
    assert diag["qualifying"] == 1

    trades, _ = backtest_engine.run_backtest(
        frame, _factory(frame, events={TICKER: [spun]}))
    assert len(trades) == 0
    trades, _ = backtest_engine.run_backtest(
        frame, _factory(frame, events={TICKER: [cash]}))
    assert len(trades) == 1


@pytest.mark.parametrize("factor", [1.319, 0.5, 2.0, 3.0, 1.02])
def test_any_non_unit_split_factor_excludes_the_event(factor):
    frame = _frame()
    record = dict(_record(frame.index, ABOVE_1, 0.010), split_on_ex_date=factor)
    entries, _, diag = mod.event_signal_dates(
        frame.index, [record], mod.YIELD_FLOOR)
    assert entries == set()
    assert diag["excluded_split_type_adjustment"] == 1


def test_secondary_share_class_carries_no_events_so_an_issuer_counts_once():
    """parameters.pool pre-registers ONE share class per issuer ("so a
    single corporate event is not counted twice") but TRADED is the full
    resolved pool, so without this rule GOOG/GOOGL, FOX/FOXA and NWS/NWSA
    would each contribute the same dividends twice. TRADED and PARAMETERS are
    pre-registered and are not edited; the secondary class instead carries no
    events."""
    assert mod.SECONDARY_SHARE_CLASSES == ("GOOG", "FOX", "NWS")
    frame = _frame()
    records = _events(frame.index)[TICKER]
    for secondary, primary in (("GOOG", "GOOGL"), ("FOX", "FOXA"),
                               ("NWS", "NWSA")):
        events = {secondary: records, primary: records}
        frames = {secondary: frame, primary: frame}
        dropped = mod.make_signal_fn(secondary, frame.index, frames,
                                     events=events)
        kept = mod.make_signal_fn(primary, frame.index, frames, events=events)
        assert dropped.event_counts["excluded_secondary_share_class"] is True
        assert dropped.event_counts[
            "events_dropped_secondary_share_class"] == len(records)
        assert kept.event_counts["excluded_secondary_share_class"] is False
        trades_dropped, _ = backtest_engine.run_backtest(frame, dropped)
        trades_kept, _ = backtest_engine.run_backtest(frame, kept)
        assert len(trades_dropped) == 0
        assert len(trades_kept) == len(FIRING)


# ------------------------------------------------------- fetcher: basis check

def _split_frame(*, ratio: float, applied: bool) -> pd.DataFrame:
    """20 sessions with a split of ``ratio`` reported on session 10.

    ``applied=True`` mimics yfinance having already divided the pre-split
    closes by the ratio, so the series is continuous across the split (step
    1.0). ``applied=False`` leaves the pre-split closes on the raw basis, so
    the 1/ratio step is still there -- the footprint of a split that reached
    the actions/dividend series but not the price history."""
    index = pd.bdate_range("2026-08-03", periods=20)
    closes = np.full(20, 100.0)
    if not applied:
        closes[:10] = 100.0 * ratio
    splits = np.zeros(20)
    splits[10] = ratio
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes,
                         "Close": closes, "Dividends": np.zeros(20),
                         "Stock Splits": splits}, index=index)


def test_fetcher_basis_check_flags_an_unapplied_split_and_passes_an_applied_one():
    """The APH failure mode, unit-tested: a split present in the actions /
    dividend series but not in the price history leaves the raw 1/ratio step
    behind, and every yield for that member is then mis-scaled."""
    import fetch_batch5_events_ex_dividend_day_capture_pooled as fetcher

    summary, violations = fetcher.verify_split_basis(
        "APH", _split_frame(ratio=2.0, applied=False))
    assert summary["splits_material"] == 1
    assert summary["price_unapplied"] == 1
    assert len(violations) == 1
    assert violations[0]["ticker"] == "APH"
    assert violations[0]["kind"] == "price_history_unapplied"
    assert violations[0]["split_ratio"] == 2.0
    assert violations[0]["split_session"] == "2026-08-17"

    summary, violations = fetcher.verify_split_basis(
        "APH", _split_frame(ratio=2.0, applied=True))
    assert summary["splits_material"] == 1
    assert summary["price_applied"] == 1
    assert violations == []


def test_fetcher_basis_check_abstains_on_an_undecidable_ratio():
    """A 1.02 stock dividend is indistinguishable from an ordinary -2% day, so
    the check declines to decide rather than manufacturing a violation."""
    import fetch_batch5_events_ex_dividend_day_capture_pooled as fetcher
    summary, violations = fetcher.verify_split_basis(
        "HON", _split_frame(ratio=1.02, applied=False))
    assert summary["splits_seen"] == 1
    assert summary["splits_material"] == 0
    assert summary["splits_immaterial_ratio"] == 1
    assert violations == []


# ------------------------------------- fetcher: the REVERSE basis direction
# A PRICE-side-only check (the schema-2 shape) can only ever see the Amphenol
# order -- dividends adjusted, prices lagging -- which UNDERstates a yield and
# is conservative. The reverse order (prices adjusted, amounts still
# pre-split) leaves no step in the price series at all, so such a check
# passes it silently, while it OVERstates every pre-split yield by the split
# ratio and pushes events above the floor.

def _reverse_frame(*, ratio: float, dividend_basis: str, n: int = 40,
                   split_at: int = 20, dividends_after: bool = True,
                   post_amount: float = 0.50) -> pd.DataFrame:
    """Prices already split-adjusted (a flat, continuous series across the
    split session), with the dividend series on ``dividend_basis``:

    ``"rebased"`` -- the vendor divided the pre-split amounts by the ratio,
    so the series steps smoothly (the healthy case).
    ``"raw"``     -- the pre-split amounts are still on the pre-split basis,
    so they sit a factor ``ratio`` above their post-split neighbours. That is
    direction B, and it is invisible in the price series.
    """
    index = pd.bdate_range("2026-01-05", periods=n)
    closes = np.full(n, 100.0)
    splits = np.zeros(n)
    splits[split_at] = ratio
    dividends = np.zeros(n)
    before = post_amount * (ratio if dividend_basis == "raw" else 1.0)
    for pos in (split_at - 16, split_at - 11, split_at - 6, split_at - 1):
        dividends[pos] = before
    if dividends_after:
        for pos in (split_at + 4, split_at + 9, split_at + 14, split_at + 19):
            if pos < n:
                dividends[pos] = post_amount
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes,
                         "Close": closes, "Dividends": dividends,
                         "Stock Splits": splits}, index=index)


def test_fetcher_basis_check_detects_the_reverse_race_the_price_side_cannot_see():
    """Direction B: prices already split-adjusted while the amounts are still
    on the pre-split basis. The price step across the split is an ordinary
    1.0, so a price-side-only check passes silently -- yet every pre-split
    yield is OVERSTATED by the ratio, the anti-conservative direction."""
    import fetch_batch5_events_ex_dividend_day_capture_pooled as fetcher

    summary, violations = fetcher.verify_split_basis(
        "SYN", _reverse_frame(ratio=2.0, dividend_basis="raw"))
    # the price side sees nothing at all: that is the whole point
    assert summary["price_applied"] == 1
    assert summary["price_unapplied"] == 0
    assert summary["dividend_unadjusted"] == 1
    assert [v["kind"] for v in violations] == ["dividend_series_unadjusted"]
    assert violations[0]["dividend_side_step"] == pytest.approx(2.0)
    assert violations[0]["price_side_reading"] == "applied"

    summary, violations = fetcher.verify_split_basis(
        "SYN", _reverse_frame(ratio=2.0, dividend_basis="rebased"))
    assert summary["dividend_refuted_raw_basis"] == 1
    assert summary["dividend_unadjusted"] == 0
    assert violations == []


def test_fetcher_basis_check_does_not_read_direction_b_on_a_value_derived_factor():
    """PNR 2018-05-01 (the nVent spin-off): factor 1.489, both series
    correctly on one basis, and Pentair's genuine post-spin dividend cut
    (0.35 -> 0.175 raw) puts the observed step at 0.235057/0.175 = 1.34 --
    inside a band centred on 1.489. Reading direction B there would call that
    a violation and refuse to write the whole table. Nothing ties an issuer's
    dividend change to a value-derived factor, so direction B is not read
    there; it is counted and disclosed instead."""
    import fetch_batch5_events_ex_dividend_day_capture_pooled as fetcher
    frame = _reverse_frame(ratio=1.489, dividend_basis="rebased",
                           post_amount=0.175)
    frame.loc[frame["Dividends"] > 0, "Dividends"] = [
        0.235057, 0.235057, 0.235057, 0.235057, 0.175, 0.175, 0.175, 0.175]
    summary, violations = fetcher.verify_split_basis("PNR", frame)
    assert violations == []
    assert summary["dividend_not_read_value_derived_factor"] == 1
    assert summary["dividend_unadjusted"] == 0
    # and the events it could not clear are counted as exposure, not ignored
    assert summary["events_before_an_undecided_split"] == 4
    # the same step under a genuine 1.5-for-1 split IS read
    assert fetcher.is_announced_share_ratio(1.489) is False
    assert fetcher.is_announced_share_ratio(1.5) is True


def test_fetcher_basis_check_is_blind_to_a_split_with_no_dividends_after_it():
    """The honest limit, stated as a test: a split too recent for two
    dividends to have been paid since is exactly when a propagation race is in
    flight, and direction B rescales every pre-split amount uniformly, so the
    member's own data carries no contrast that could expose it. The reading
    must report that gap, not pass."""
    import fetch_batch5_events_ex_dividend_day_capture_pooled as fetcher
    summary, violations = fetcher.verify_split_basis(
        "SYN", _reverse_frame(ratio=2.0, dividend_basis="raw",
                              dividends_after=False))
    assert violations == []
    assert summary["dividend_undecidable_too_few_after"] == 1
    assert summary["events_before_an_undecided_split"] == 4


def test_basis_check_record_names_both_directions_and_its_own_limits():
    import fetch_batch5_events_ex_dividend_day_capture_pooled as fetcher
    summary, _ = fetcher.verify_split_basis(
        "SYN", _reverse_frame(ratio=2.0, dividend_basis="rebased"))
    record = fetcher.basis_check_record(summary)
    assert set(record["directions"]) == set(mod.REQUIRED_BASIS_DIRECTIONS)
    assert record["status"] == "verified"
    assert record["violations"] == []
    assert "does_not_prove" in record and record["does_not_prove"]
    mod.validate_events_payload(_payload(basis_check=record))


def test_fetcher_records_are_causal_and_carry_the_mechanism_evidence():
    """Every evidence field must be computable from data <= the ex-date, so a
    downstream filter using one cannot become look-ahead."""
    import fetch_batch5_events_ex_dividend_day_capture_pooled as fetcher
    index = pd.bdate_range("2026-01-05", periods=30)
    closes = np.full(30, 50.0)
    dividends = np.zeros(30)
    for position, amount in ((5, 0.10), (11, 0.10), (17, 0.12), (23, 0.10),
                             (29, 5.00)):
        dividends[position] = amount
    splits = np.zeros(30)
    splits[23] = 1.5
    frame = pd.DataFrame({"Open": closes, "High": closes, "Low": closes,
                          "Close": closes, "Dividends": dividends,
                          "Stock Splits": splits}, index=index)
    records = fetcher.build_records(frame)
    assert [r["ex_date"] for r in records] == [
        index[position].strftime("%Y-%m-%d") for position in (5, 11, 17, 23, 29)]
    assert records[0]["prior_amount"] is None
    assert records[0]["prior_count"] == 0
    assert records[0]["trailing_median_amount"] is None
    assert records[3]["split_on_ex_date"] == 1.5
    assert records[4]["split_on_ex_date"] == 0.0
    assert records[4]["prior_amount"] == 0.10
    assert records[4]["prior_count"] == 4
    assert records[4]["trailing_median_amount"] == pytest.approx(0.10)
    assert records[4]["signal_date"] == index[27].strftime("%Y-%m-%d")
    assert records[4]["unadjusted_close_signal"] == 50.0
    assert records[4]["yield"] == pytest.approx(0.1)
