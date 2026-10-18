import os
import sys

import numpy as np
import pandas as pd
import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining.abi1_driver import (build_pooled_curve, build_rotation_curve,
                                build_single_asset_curve, evaluate_family,
                                record_abi1_outcome)
from mining import account_relative, benchmark_relative, constants, evidence, trial_ledger
from mining.backtest_engine import SignalResult


# --------------------------------------------------------- single-asset shape

def _single_asset_frame():
    # 4 sessions. Open/Close chosen so entry_px != Close on the entry bar
    # and every bar contributes a distinct, nonzero price return -- a
    # trivial (Open == Close) fixture would make the entry-day leg of the
    # compounding identity a no-op factor and could hide a bug.
    index = pd.date_range("2024-01-02", periods=4, freq="B")
    opens = [100.0, 100.0, 104.0, 108.0]
    closes = [100.0, 104.0, 108.0, 110.0]
    return pd.DataFrame(
        {"Open": opens, "High": closes, "Low": opens, "Close": closes},
        index=index)


def _single_asset_signal(frame, i):
    # Entry evaluated one bar ahead of the fill (run_backtest convention):
    # this fires next-open entry at index 1, signal-exit at index 3, so
    # the trade spans three return-bearing bars (1, 2, 3).
    return SignalResult(entry=(i == 0), exit=(i == 3))


def test_build_single_asset_curve_returns_aligned_finite_series():
    df = _single_asset_frame()
    curve = build_single_asset_curve(df, _single_asset_signal, cost_bps=0.0)

    assert set(curve.keys()) == {"strategy_daily", "in_market", "trades_df"}
    strategy_daily = curve["strategy_daily"]
    in_market = curve["in_market"]

    assert isinstance(strategy_daily, pd.Series)
    pd.testing.assert_index_equal(strategy_daily.index, df.index)
    assert np.isfinite(strategy_daily.to_numpy()).all()

    assert isinstance(in_market, pd.Series)
    pd.testing.assert_index_equal(in_market.index, df.index)
    assert in_market.dtype == bool

    assert len(curve["trades_df"]) == 1


def test_build_single_asset_curve_cost_zero_compounding_matches_net_ret_exactly():
    df = _single_asset_frame()
    curve = build_single_asset_curve(df, _single_asset_signal, cost_bps=0.0)
    trades_df = curve["trades_df"]
    assert len(trades_df) == 1
    row = trades_df.iloc[0]

    span = curve["strategy_daily"].loc[row["entry_date"]:row["exit_date"]]
    compounded = float((1.0 + span).prod() - 1.0)

    # At cost_bps=0, net_ret == gross_ret; compounding the daily mtm
    # series over the trade's span must reproduce it to 1e-9 -- this
    # proves the two calls share the same price-return mechanics.
    assert compounded == pytest.approx(float(row["net_ret"]), abs=1e-9)
    assert compounded == pytest.approx(float(row["gross_ret"]), abs=1e-9)


def test_build_single_asset_curve_nonzero_cost_bps_is_wired_into_mtm_exit_day():
    # mark_to_market_daily_returns charges the round-trip cost on the exit
    # session as cost / G, where G is the trade's growth over the sessions
    # before the exit session, so the compounded daily path over the trade
    # equals run_backtest's own net_ret (gross_ret - cost on entry capital)
    # exactly, including for a trade that spans more than one bar (as here).
    df = _single_asset_frame()
    cost_bps = 100.0
    cost = cost_bps / 1e4

    curve_zero = build_single_asset_curve(df, _single_asset_signal, cost_bps=0.0)
    curve_costed = build_single_asset_curve(
        df, _single_asset_signal, cost_bps=cost_bps)

    row = curve_costed["trades_df"].iloc[0]
    exit_date = curve_zero["trades_df"].iloc[0]["exit_date"]
    assert exit_date == row["exit_date"]

    # The engine's own net_ret reflects the cost once, on entry capital.
    zero_gross = float(curve_zero["trades_df"].iloc[0]["gross_ret"])
    costed_net = float(row["net_ret"])
    assert costed_net == pytest.approx(zero_gross - cost, abs=1e-9)

    # The compounded daily path over the trade reproduces that net return.
    span = curve_costed["strategy_daily"].loc[row["entry_date"]:row["exit_date"]]
    compounded = float((1.0 + span).prod() - 1.0)
    assert compounded == pytest.approx(costed_net, abs=1e-12)

    # cost_bps reached the exit session only: the earlier sessions are
    # untouched, and the exit session is charged cost / G.
    zero_span = curve_zero["strategy_daily"].loc[row["entry_date"]:row["exit_date"]]
    before = zero_span.iloc[:-1]
    pd.testing.assert_series_equal(span.iloc[:-1], before)
    growth_before_exit = float((1.0 + before).prod())
    diff = span.iloc[-1] - zero_span.iloc[-1]
    assert diff == pytest.approx(-cost / growth_before_exit, abs=1e-12)
    assert growth_before_exit != pytest.approx(1.0)   # a multi-bar trade, so G matters


# ----------------------------------------------------------- rotation shape

def _rotation_frame(opens, closes, index):
    opens = np.asarray(opens, dtype=float)
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "Open": opens,
            "High": np.maximum(opens, closes),
            "Low": np.minimum(opens, closes),
            "Close": closes,
        },
        index=index)


def _rotation_asset_frames():
    # 6 sessions, BOTH legs actually trade (not just one) -- a fixture
    # where only one asset is ever selected cannot distinguish "filtered
    # per asset" from "fed every trade to every leg", since the untraded
    # leg's trades_df is empty either way.
    index = pd.date_range("2024-01-02", periods=6, freq="B")
    risk = _rotation_frame(
        [100.0, 100.0, 102.0, 104.0, 106.0, 108.0],
        [100.0, 102.0, 104.0, 106.0, 108.0, 110.0],
        index)
    safe = _rotation_frame(
        [50.0, 50.0, 49.0, 48.0, 47.0, 46.0],
        [50.0, 49.0, 48.0, 47.0, 46.0, 45.0],
        index)
    return {"RISK": risk, "SAFE": safe}


def _rotation_target_fn(decision_frames):
    # decision_idx = len(...) - 1. RISK entered at decision_idx 0
    # (execution bar 1); cash at decision_idx 1-2 (closes RISK at
    # execution bar 2, stays flat through execution bar 3 -- bar 3 is
    # therefore a cash session untouched by any trade's entry or exit
    # day, unlike bar 2, RISK's own exit day, which legitimately carries
    # a nonzero mtm contribution and would be the wrong session to assert
    # zero on); SAFE entered at decision_idx 3 and held through
    # decision_idx 4 (execution bars 4-5, liquidated at the final close).
    n = len(decision_frames["RISK"])
    if n == 1:
        return "RISK"
    if n in (2, 3):
        return None
    return "SAFE"


def test_build_rotation_curve_returns_aligned_series_and_expected_keys():
    frames = _rotation_asset_frames()
    curve = build_rotation_curve(
        frames, _rotation_target_fn,
        cost_bps_by_asset={"RISK": 0.0, "SAFE": 0.0})

    assert set(curve.keys()) == {"strategy_daily", "in_market", "trades_df"}
    index = frames["RISK"].index
    pd.testing.assert_index_equal(curve["strategy_daily"].index, index)
    pd.testing.assert_index_equal(curve["in_market"].index, index)
    assert curve["in_market"].dtype == bool
    assert np.isfinite(curve["strategy_daily"].to_numpy()).all()
    assert len(curve["trades_df"]) == 2  # one RISK segment, one SAFE segment


def test_build_rotation_curve_cash_session_is_flat_and_out_of_market():
    frames = _rotation_asset_frames()
    curve = build_rotation_curve(
        frames, _rotation_target_fn,
        cost_bps_by_asset={"RISK": 0.0, "SAFE": 0.0})

    cash_session = frames["RISK"].index[3]
    assert curve["in_market"].loc[cash_session] == False  # noqa: E712
    assert curve["strategy_daily"].loc[cash_session] == 0.0

    # Sanity: the designated cash session really is disjoint from any
    # recorded trade's entry/exit day -- otherwise this test would pass
    # vacuously (mark_to_market_daily_returns legitimately charges a
    # nonzero return on a trade's own exit day even though in_market goes
    # False starting that same day).
    trades_df = curve["trades_df"]
    assert cash_session not in set(trades_df["entry_date"])
    assert cash_session not in set(trades_df["exit_date"])


def test_build_rotation_curve_in_market_includes_exit_session():
    frames = _rotation_asset_frames()
    curve = build_rotation_curve(
        frames, _rotation_target_fn,
        cost_bps_by_asset={"RISK": 0.0, "SAFE": 0.0})

    # bar2 exits at the open and still owns the overnight return; bar3 cash.
    assert list(curve["in_market"]) == [
        False, True, True, False, True, True]


def test_build_rotation_curve_strategy_daily_is_each_legs_OWN_priced_return():
    # Guards the per-asset filter directly: with cost_bps=0, RISK's entry
    # day (bar1) is a session where ONLY the RISK leg has an open
    # position, so its contribution must be RISK's own Close/Open return
    # on that bar -- 102/100 - 1 = 0.02 exactly. A driver that fed BOTH
    # legs' trades into every mark_to_market_daily_returns call (ignoring
    # the "asset" column) would ALSO evaluate the SAFE trade's dates
    # against RISK's frame and vice versa; SAFE's own bar1 return
    # (49/50 - 1 = -0.02) would land on this exact same session through
    # that cross-contamination and cancel the correct value to 0.0 --
    # this fixture's numbers were chosen so that failure mode is not
    # silently invisible.
    frames = _rotation_asset_frames()
    curve = build_rotation_curve(
        frames, _rotation_target_fn,
        cost_bps_by_asset={"RISK": 0.0, "SAFE": 0.0})

    risk_entry_day = frames["RISK"].index[1]
    assert curve["strategy_daily"].loc[risk_entry_day] == pytest.approx(
        102.0 / 100.0 - 1.0, abs=1e-9)


def test_build_rotation_curve_trims_both_legs_to_their_common_inception():
    # A real batch5 rotation family can have a leg that
    # lists later than the other (a real-world recently-listed-instrument
    # shape, the same one pooled families already guard against).
    # build_rotation_curve must not raise on a calendar mismatch; it must
    # instead trim BOTH legs to their common inception first,
    # matching run_batch3_mining._normalize_rotation_inception's own
    # precedent (adapted for full history -- no windowing here).
    frames = _rotation_asset_frames()
    late_safe = frames["SAFE"].iloc[2:].copy()  # SAFE lists 2 sessions later
    trimmed_frames = {"RISK": frames["RISK"], "SAFE": late_safe}

    curve = build_rotation_curve(
        trimmed_frames, lambda _: None,
        cost_bps_by_asset={"RISK": 0.0, "SAFE": 0.0})

    expected_index = frames["RISK"].index[2:]
    pd.testing.assert_index_equal(curve["strategy_daily"].index, expected_index)
    pd.testing.assert_index_equal(curve["in_market"].index, expected_index)
    assert len(curve["trades_df"]) == 0


def test_build_rotation_curve_raises_on_a_genuine_interior_calendar_mismatch():
    # A prefix difference (common inception) is trimmed away silently; an
    # INTERIOR gap between the two traded legs after that trim is a real
    # data defect and must fail closed, not be silently intersected away
    # (run_batch3_mining._rotation_bucket's own precedent: traded legs must
    # match EXACTLY, only signal-only frames get intersected).
    frames = _rotation_asset_frames()
    gappy_safe = frames["SAFE"].drop(frames["SAFE"].index[3])
    with pytest.raises(ValueError, match="identical calendar"):
        build_rotation_curve(
            {"RISK": frames["RISK"], "SAFE": gappy_safe}, lambda _: None,
            cost_bps_by_asset={"RISK": 0.0, "SAFE": 0.0})


def test_build_rotation_curve_signal_frames_reach_the_target_fn_truncated():
    # A real batch5 rotation family
    # (month_end_rebalancing_flow_rotation) declares a signal-only ticker
    # (IEF) and its make_target_fn closure reads it straight out of the
    # engine's own `truncated` dict (`truncated["IEF"]`) -- without a
    # `signal_frames` parameter on build_rotation_curve,
    # run_two_asset_rotation would have nothing to put there and a
    # real production family would KeyError. This pins that: a signal
    # frame reaches target_fn, truncated to the SAME decision bar as the
    # traded legs, and the account curve is aligned to the intersection of
    # every frame's own calendar (traded legs + signal frames).
    frames = _rotation_asset_frames()
    sig = frames["RISK"].iloc[2:].copy()  # signal-only, starts later
    calls = []

    def target_fn(decision_frames):
        calls.append(set(decision_frames))
        assert len(decision_frames["SIG"]) == len(decision_frames["RISK"])
        return None

    curve = build_rotation_curve(
        {"RISK": frames["RISK"], "SAFE": frames["SAFE"]}, target_fn,
        cost_bps_by_asset={"RISK": 0.0, "SAFE": 0.0},
        signal_frames={"SIG": sig})

    assert calls and all("SIG" in keys for keys in calls)
    expected_index = frames["RISK"].index[2:]
    pd.testing.assert_index_equal(curve["strategy_daily"].index, expected_index)
    pd.testing.assert_index_equal(curve["in_market"].index, expected_index)


# ------------------------------------------------------------- pooled shape

def _pooled_member_frame(index, base):
    # Deterministic, distinct nonzero daily returns per bar -- a flat
    # frame would make in_market meaningful but strategy_daily trivially
    # zero everywhere, which could hide a bug in the mean-with-zero-fill wiring.
    n = len(index)
    pct = np.array([0.012 if k % 2 == 0 else -0.007 for k in range(n)])
    closes = base * np.cumprod(1.0 + pct)
    opens = np.empty(n)
    opens[0] = base
    opens[1:] = closes[:-1]
    highs = np.maximum(opens, closes)
    lows = np.minimum(opens, closes)
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes},
        index=index)


def _cadence_signal(entry_mod, exit_mod, period):
    # A different cadence per member (not one function shared across the
    # pool) both matches the real production wiring (each member gets its
    # OWN signal function -- see build_pooled_curve's own docstring) and
    # makes the fixture properly discriminating: a bug that mixed up which
    # signal function belongs to which ticker would produce different
    # trade dates than the correct pairing.
    def signal(frame, i):
        return SignalResult(entry=(i % period == entry_mod),
                            exit=(i % period == exit_mod))
    return signal


_SIGNAL_A = _cadence_signal(0, 3, 10)
_SIGNAL_B = _cadence_signal(0, 2, 7)
_SIGNAL_C = _cadence_signal(0, 5, 13)


def _pooled_fixture():
    # Three members on THREE DIFFERENT, STAGGERED calendars -- A covers
    # sessions 1-40, B sessions 31-70, C sessions 61-100 -- so no single
    # member's own mask ever equals the pool's true OR over the full union
    # calendar (A has no bar at all for sessions 41-100, etc.), and the
    # union-calendar/reindex/zero-fill path is unavoidably exercised. This
    # is the same shape test_mining_pooling.py's own
    # test_pooled_daily_returns_sums_overlapping_exit_days_across_tickers
    # fixture uses (non-overlapping-start date ranges), for the same reason:
    # recently-listed universe members have shorter calendars, and
    # pooled-family coverage must still hold across them.
    #
    # Exclusive-coverage regions this staggering guarantees (global session
    # index, 0-based): A-only 0-29, A/B overlap 30-39, B-only 40-59, B/C
    # overlap 60-69, C-only 70-99. Union of all three = the full 100-session
    # calendar (verified by a test below, not just asserted here).
    full_calendar = pd.date_range("2024-01-02", periods=100, freq="B")
    full_frames = {
        "A": _pooled_member_frame(full_calendar[0:40], base=100.0),
        "B": _pooled_member_frame(full_calendar[30:70], base=50.0),
        "C": _pooled_member_frame(full_calendar[60:100], base=200.0),
    }
    per_ticker_signal_fn = {"A": _SIGNAL_A, "B": _SIGNAL_B, "C": _SIGNAL_C}
    cost_bps_by_symbol = {"A": 0.0, "B": 0.0, "C": 0.0}
    return full_calendar, full_frames, per_ticker_signal_fn, cost_bps_by_symbol


def _reference_member_curves(full_frames, per_ticker_signal_fn, cost_bps_by_symbol):
    # Independent per-member curves, computed via the SAME
    # build_single_asset_curve the driver uses internally -- used here only
    # to derive an expected pooled result via straightforward pandas
    # reindex/OR/mean logic in the test itself (a different code path than
    # build_pooled_curve's own combination step), not to re-exercise the
    # implementation under test.
    return {
        symbol: build_single_asset_curve(
            full_frames[symbol], per_ticker_signal_fn[symbol],
            cost_bps=float(cost_bps_by_symbol[symbol]))
        for symbol in full_frames
    }


def test_build_pooled_curve_returns_aligned_finite_series_on_the_union_calendar():
    full_calendar, full_frames, per_ticker_signal_fn, cost_bps_by_symbol = _pooled_fixture()
    curve = build_pooled_curve(
        per_ticker_signal_fn, full_frames, cost_bps_by_symbol=cost_bps_by_symbol)

    assert set(curve.keys()) == {"strategy_daily", "in_market", "trades_df"}

    union = (full_frames["A"].index.union(full_frames["B"].index)
             .union(full_frames["C"].index).sort_values())
    pd.testing.assert_index_equal(curve["strategy_daily"].index, union)
    pd.testing.assert_index_equal(curve["in_market"].index, union)
    assert curve["in_market"].dtype == bool
    assert np.isfinite(curve["strategy_daily"].to_numpy()).all()

    # Confirms the fixture's own staggering is what this test thinks it is
    # (a full, gapless 100-session union), not an accidental full overlap
    # or a hidden gap.
    pd.testing.assert_index_equal(union, full_calendar)


def test_build_pooled_curve_in_market_matches_manual_or_across_staggered_members():
    full_calendar, full_frames, per_ticker_signal_fn, cost_bps_by_symbol = _pooled_fixture()
    reference = _reference_member_curves(full_frames, per_ticker_signal_fn, cost_bps_by_symbol)
    curve = build_pooled_curve(
        per_ticker_signal_fn, full_frames, cost_bps_by_symbol=cost_bps_by_symbol)

    union = curve["in_market"].index
    expected = pd.Series(False, index=union)
    for member_curve in reference.values():
        expected = expected | member_curve["in_market"].reindex(union, fill_value=False)
    pd.testing.assert_series_equal(curve["in_market"], expected, check_names=False)

    # A-only session (global index 5): B's frame starts at global 30, C's
    # at global 60, so ONLY A has a bar here at all -- exercises the
    # reindex/fill_value=False path directly (B/C must contribute nothing,
    # not a spurious True) and would catch a buggy implementation that just
    # forwarded one dominant member's own in_market mask instead of a real
    # union.
    a_only_session = full_calendar[5]
    assert a_only_session not in full_frames["B"].index
    assert a_only_session not in full_frames["C"].index
    assert (curve["in_market"].loc[a_only_session]
            == reference["A"]["in_market"].loc[a_only_session])


def test_build_pooled_curve_strategy_daily_matches_manual_mean_with_full_count_divisor():
    full_calendar, full_frames, per_ticker_signal_fn, cost_bps_by_symbol = _pooled_fixture()
    reference = _reference_member_curves(full_frames, per_ticker_signal_fn, cost_bps_by_symbol)
    curve = build_pooled_curve(
        per_ticker_signal_fn, full_frames, cost_bps_by_symbol=cost_bps_by_symbol)

    union = curve["strategy_daily"].index
    expected = pd.Series(0.0, index=union)
    for member_curve in reference.values():
        expected = expected.add(
            member_curve["strategy_daily"].reindex(union, fill_value=0.0), fill_value=0.0)
    expected = expected / 3.0  # full member count, not the count present that session

    np.testing.assert_allclose(
        curve["strategy_daily"].to_numpy(), expected.to_numpy(), atol=1e-9)

    # Three hand-picked sessions, one from each member's EXCLUSIVE coverage
    # window (A-only: global 0-29; B-only: 40-59; C-only: 70-99) -- each
    # must divide by the FULL pool size (3), not the 1 member actually
    # present that session. The full-array check above already covers this,
    # but these three sessions pin the exact divisor directly against each
    # member's own reference value, independent of the array-wide compare.
    for symbol, session in (("A", full_calendar[5]), ("B", full_calendar[45]),
                            ("C", full_calendar[80])):
        manual = reference[symbol]["strategy_daily"].loc[session] / 3.0
        assert curve["strategy_daily"].loc[session] == pytest.approx(manual, abs=1e-9)


def test_build_pooled_curve_trades_df_row_count_equals_sum_of_member_trade_counts():
    full_calendar, full_frames, per_ticker_signal_fn, cost_bps_by_symbol = _pooled_fixture()
    reference = _reference_member_curves(full_frames, per_ticker_signal_fn, cost_bps_by_symbol)
    curve = build_pooled_curve(
        per_ticker_signal_fn, full_frames, cost_bps_by_symbol=cost_bps_by_symbol)

    expected_count = sum(len(c["trades_df"]) for c in reference.values())
    assert expected_count > 0  # sanity: the fixture actually produces trades
    assert len(curve["trades_df"]) == expected_count
    assert set(curve["trades_df"]["ticker"]) == {"A", "B", "C"}
    for symbol, member_curve in reference.items():
        assert (curve["trades_df"]["ticker"] == symbol).sum() == len(member_curve["trades_df"])


def test_build_pooled_curve_refuses_an_empty_pool():
    with pytest.raises(ValueError, match="empty"):
        build_pooled_curve({}, {}, cost_bps_by_symbol={})


# --------------------------------------------------------- evaluate_family

# 300 > account_relative.MIN_ACCOUNT_SESSIONS (252), so layer 1 actually
# scores instead of refusing on session count; 300 <
# constants.ABI1_HOLDOUT_SESSIONS (756) + 252, so the holdout split is
# refused and layer 1 scores the FULL window -- deliberate, so this fixture
# does not also have to satisfy the 3-year holdout floor to exercise the
# wiring under test.
_EVAL_FAMILY_SESSIONS = 300


def _evaluate_family_frame(n, base):
    # Same deterministic, distinct-nonzero-daily-move construction as
    # _pooled_member_frame above -- a flat frame would make strategy_daily
    # trivially zero and hide a real wiring bug.
    index = pd.date_range("2020-01-02", periods=n, freq="B")
    pct = np.array([0.004 if k % 3 == 0 else -0.0015 for k in range(n)])
    closes = base * np.cumprod(1.0 + pct)
    opens = np.empty(n)
    opens[0] = base
    opens[1:] = closes[:-1]
    highs = np.maximum(opens, closes)
    lows = np.minimum(opens, closes)
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes},
        index=index)


def _evaluate_family_curve(n=_EVAL_FAMILY_SESSIONS):
    frame = _evaluate_family_frame(n, base=100.0)

    def signal(frame, i):
        # One trade spanning nearly the whole window: next-open entry at
        # bar 1, signal-exit at the LAST bar (same run_backtest convention
        # as _single_asset_signal above) -- so trades_df is non-empty and
        # tax_policy="realization" has something real to net.
        return SignalResult(entry=(i == 0), exit=(i == n - 1))

    curve = build_single_asset_curve(frame, signal, cost_bps=0.0)
    assert len(curve["trades_df"]) == 1  # fixture sanity
    return curve


def _evaluate_family_benchmark_frame(n=_EVAL_FAMILY_SESSIONS):
    # A DIFFERENT deterministic series (own base price, own cadence) on the
    # SAME date index as the curve fixture -- covering the same full span,
    # without being byte-identical to the strategy series (which would make
    # a benchmark-wiring bug invisible).
    return _evaluate_family_frame(n, base=300.0)


def test_evaluate_family_returns_the_documented_evaluate_shape():
    curve = _evaluate_family_curve()
    benchmark_frame = _evaluate_family_benchmark_frame()

    result = evaluate_family(
        curve, benchmark_frame=benchmark_frame, benchmark_entry_cost_bps=0.0,
        account_type="taxable", tax_policy="realization")

    assert result["construction"] == "ABI-1"
    assert result["verdict"] in account_relative.VERDICTS
    assert result["tax_policy"] == "realization"
    assert set(result["layer1"].keys()) >= {"scored", "dsr_value", "reason"}
    assert result["layer1"]["scored"] is True  # 300 sessions clears the floor
    assert set(result["layer2"].keys()) >= {"all_refusals_passed"}
    assert "evaluated" in result["holdout"] and "passed" in result["holdout"]
    assert isinstance(result["account_daily"], pd.Series)
    assert isinstance(result["log_excess"], pd.Series)
    # account_type="taxable" must actually reach evaluate() as the STCG
    # rate, not get lost or silently hardcoded to some other constant --
    # this pins that wiring.
    assert result["tax_haircut_rate"] == pytest.approx(constants.STCG_TAX_RATE)
    assert result["ltcg_tax_rate"] == pytest.approx(constants.LTCG_TAX_RATE)


def test_evaluate_family_tax_policy_none_zeroes_both_rates_regardless_of_account_type():
    curve = _evaluate_family_curve()
    benchmark_frame = _evaluate_family_benchmark_frame()

    result = evaluate_family(
        curve, benchmark_frame=benchmark_frame, benchmark_entry_cost_bps=0.0,
        account_type="taxable", tax_policy="none")

    assert result["tax_policy"] == "none"
    assert result["tax_haircut_rate"] == 0.0
    assert result["ltcg_tax_rate"] == 0.0
    assert result["tax_loss_netting"] is False


def test_evaluate_family_tax_advantaged_account_gets_zero_haircut_under_realization():
    curve = _evaluate_family_curve()
    benchmark_frame = _evaluate_family_benchmark_frame()

    result = evaluate_family(
        curve, benchmark_frame=benchmark_frame, benchmark_entry_cost_bps=0.0,
        account_type="roth", tax_policy="realization")

    # cost_tax_inputs.tax_haircut_rate_for("roth") == 0.0 -- a tax-advantaged
    # account must not be hardcoded to STCG_TAX_RATE.
    assert result["tax_policy"] == "realization"
    assert result["tax_haircut_rate"] == 0.0
    assert result["ltcg_tax_rate"] == pytest.approx(constants.LTCG_TAX_RATE)


def test_evaluate_family_rejects_unknown_tax_policy():
    curve = _evaluate_family_curve()
    benchmark_frame = _evaluate_family_benchmark_frame()

    with pytest.raises(ValueError, match="tax_policy"):
        evaluate_family(
            curve, benchmark_frame=benchmark_frame,
            benchmark_entry_cost_bps=0.0, account_type="taxable",
            tax_policy="mark_to_market")


def test_evaluate_family_rejects_unknown_account_type_under_realization():
    # account_type must really reach cost_tax_inputs.tax_haircut_rate_for,
    # not get swallowed -- that function raises on an unrecognized
    # account_type rather than silently defaulting.
    curve = _evaluate_family_curve()
    benchmark_frame = _evaluate_family_benchmark_frame()

    with pytest.raises(ValueError, match="account_type"):
        evaluate_family(
            curve, benchmark_frame=benchmark_frame,
            benchmark_entry_cost_bps=0.0, account_type="checking",
            tax_policy="realization")


def test_evaluate_family_scores_a_curve_with_no_account_kind_on_the_whole_account():
    """A curve with no account_kind key (every builder except the overlay
    ones) is judged on the whole account."""
    curve = _evaluate_family_curve()
    assert "account_kind" not in curve
    result = evaluate_family(
        curve, benchmark_frame=_evaluate_family_benchmark_frame(),
        benchmark_entry_cost_bps=0.0, account_type="taxable",
        tax_policy="realization")
    assert result["layer2"]["account_basis"] == "whole_account"


def test_evaluate_family_scores_a_benchmark_plus_overlay_curve_on_the_overlay_only_excess():
    curve = dict(_evaluate_family_curve(),
                 account_kind=account_relative.ACCOUNT_KIND_BENCHMARK_PLUS_OVERLAY)
    benchmark_frame = _evaluate_family_benchmark_frame()
    result = evaluate_family(
        curve, benchmark_frame=benchmark_frame, benchmark_entry_cost_bps=0.0,
        account_type="taxable", tax_policy="realization")
    assert result["layer2"]["account_basis"] == "overlay_only_excess"
    # the wiring, not just the label: the layer-2 numbers are the excess
    # account's, i.e. exactly what evaluate() returns when told the basis
    expected = account_relative.evaluate(
        strategy_daily=curve["strategy_daily"], in_market=curve["in_market"],
        benchmark_daily=benchmark_relative.buy_and_hold_daily_returns(benchmark_frame),
        cash_daily=account_relative.sweep_cash_daily(curve["strategy_daily"].index),
        n_trials=constants.ABI1_N_TRIALS, sr_variance=constants.DAILY_SR_VARIANCE,
        trades_df=curve["trades_df"], tax_policy="realization",
        tax_haircut_rate=constants.STCG_TAX_RATE, ltcg_tax_rate=constants.LTCG_TAX_RATE,
        risk_account_basis="overlay_only_excess")
    assert result["layer2"] == expected["layer2"]
    assert result["verdict"] == expected["verdict"]


def test_evaluate_family_refuses_an_unknown_account_kind_rather_than_guessing_a_basis():
    curve = dict(_evaluate_family_curve(), account_kind="hedged_overlay")
    with pytest.raises(ValueError, match="account_kind"):
        evaluate_family(
            curve, benchmark_frame=_evaluate_family_benchmark_frame(),
            benchmark_entry_cost_bps=0.0, account_type="taxable",
            tax_policy="realization")


def test_evaluate_family_zero_cost_benchmark_reproduces_a_direct_evaluate_call():
    # Proves the WIRING (not cost_adjusted_benchmark_daily_returns' own
    # zero-cost identity, which is that function's own test): drive
    # evaluate_family end to end at benchmark_entry_cost_bps=0.0 and confirm
    # it reaches the exact same account_relative.evaluate() result as
    # calling evaluate() directly with buy_and_hold_daily_returns as the
    # benchmark and sweep_cash_daily as cash -- the same three pieces
    # evaluate_family is supposed to assemble.
    curve = _evaluate_family_curve()
    benchmark_frame = _evaluate_family_benchmark_frame()

    via_driver = evaluate_family(
        curve, benchmark_frame=benchmark_frame, benchmark_entry_cost_bps=0.0,
        account_type="taxable", tax_policy="realization")

    expected = account_relative.evaluate(
        strategy_daily=curve["strategy_daily"], in_market=curve["in_market"],
        benchmark_daily=benchmark_relative.buy_and_hold_daily_returns(benchmark_frame),
        cash_daily=account_relative.sweep_cash_daily(curve["strategy_daily"].index),
        n_trials=constants.ABI1_N_TRIALS, sr_variance=constants.DAILY_SR_VARIANCE,
        trades_df=curve["trades_df"], tax_policy="realization",
        tax_haircut_rate=constants.STCG_TAX_RATE, ltcg_tax_rate=constants.LTCG_TAX_RATE)

    assert via_driver["verdict"] == expected["verdict"]
    assert via_driver["layer1"]["dsr_value"] == pytest.approx(expected["layer1"]["dsr_value"])
    pd.testing.assert_series_equal(via_driver["account_daily"], expected["account_daily"])
    pd.testing.assert_series_equal(via_driver["log_excess"], expected["log_excess"])


def test_evaluate_family_nonzero_benchmark_cost_actually_reaches_cost_adjusted_benchmark():
    # Confirms benchmark_entry_cost_bps really reaches
    # cost_adjusted_benchmark_daily_returns rather than being dropped or
    # applied twice. That function charges the EXACT value it is given on
    # the benchmark's first bar with no further halving inside it (its own
    # docstring: "Callers pass entry_cost_bps already halved") -- so the
    # ONLY session log_excess can differ on is the benchmark's first bar,
    # and it must move in the direction a costlier benchmark implies (the
    # benchmark looks worse, so the account's measured excess over it goes
    # UP), never on any other session.
    curve = _evaluate_family_curve()
    benchmark_frame = _evaluate_family_benchmark_frame()
    cost_bps = 12.5

    zero_cost = evaluate_family(
        curve, benchmark_frame=benchmark_frame, benchmark_entry_cost_bps=0.0,
        account_type="taxable", tax_policy="realization")
    costed = evaluate_family(
        curve, benchmark_frame=benchmark_frame, benchmark_entry_cost_bps=cost_bps,
        account_type="taxable", tax_policy="realization")

    first_session = zero_cost["log_excess"].index[0]
    assert (costed["log_excess"].loc[first_session]
            > zero_cost["log_excess"].loc[first_session])

    rest = zero_cost["log_excess"].index[1:]
    pd.testing.assert_series_equal(
        costed["log_excess"].loc[rest], zero_cost["log_excess"].loc[rest])


# ------------------------------------------------------- record_abi1_outcome

def _catalog_entry(family_id="abi1_test_family"):
    return {"family_id": family_id, "name": "x", "mechanism": "x",
            "tickers": ["PFF"], "parameters": {"lookback": 20},
            "account_types": ["taxable"], "status": "eligible",
            "source": "x", "added_date": "2026-09-01"}


def _last_outcome(ledger_path):
    rows = [r for r in trial_ledger.read_all(ledger_path)
            if r["event"] == "outcome"]
    assert len(rows) == 1, f"expected exactly one outcome row, got {len(rows)}"
    return rows[0]


def _hand_built_result(*, verdict, idx, n_sessions=444, dsr_value=0.99,
                       sharpe_daily_excess=0.07,
                       holdout_split_applied=False, holdout_sessions=0,
                       in_sample_sessions=None, n_holdout=0,
                       required_sessions=756, holdout_refusal=None,
                       reason="x"):
    """A minimal stand-in for account_relative.evaluate()'s real return
    shape, carrying only the keys record_abi1_outcome actually reads. Used
    for five of the verdicts; cut_dsr routes through a real evaluate()
    call instead (below)."""
    if in_sample_sessions is None:
        in_sample_sessions = n_sessions
    return {
        "verdict": verdict,
        "layer1": {"scored": dsr_value is not None, "dsr_value": dsr_value,
                  "sharpe_daily_excess": sharpe_daily_excess,
                  "n_sessions": n_sessions, "reason": reason},
        "account_daily": pd.Series(0.0, index=idx),
        "window": {"holdout_split_applied": holdout_split_applied,
                  "holdout_sessions": holdout_sessions,
                  "in_sample_sessions": in_sample_sessions},
        "holdout": {"n_holdout": n_holdout,
                   "required_sessions": required_sessions,
                   "refusal": holdout_refusal},
    }


def test_record_abi1_outcome_survived_writes_a_scored_row_with_ticker_label(tmp_path):
    idx = pd.bdate_range("2020-01-02", periods=300)
    result = _hand_built_result(verdict="survived", idx=idx, n_sessions=300)
    ledger = str(tmp_path / "ledger.jsonl")

    stage = record_abi1_outcome(
        ledger, "fam_survived", "run-1", "2026-09-09", result,
        catalog_entry=_catalog_entry(), instrument_set=["PFF"],
        cash_convention="sweep_cash_daily", benchmark="SPY",
        tax_policy="realization", ticker_label="pool-of-3",
        signal_source_path="scripts/run_abi1_mining.py",
        signal_source_sha256="a" * 64)

    assert stage == "survived"
    row = _last_outcome(ledger)
    assert row["family_id"] == "fam_survived"
    assert row["stage"] == "survived"
    assert row["construction"] == "ABI-1"
    assert row["dsr_value"] == pytest.approx(0.99)
    assert row["sharpe_in_sample"] == pytest.approx(0.07)
    assert row["evidence"] is None
    assert row["ticker_label"] == "pool-of-3"
    assert row["note"] == ("cash_convention=sweep_cash_daily; "
                           "benchmark=SPY; tax_policy=realization")


def test_record_abi1_outcome_extra_note_appends_to_the_scored_note_template(tmp_path):
    """A bespoke driver (scripts/run_cboe_rotation_mining.py) prices one leg
    off a disclosed assumed constant, not a real quote -- extra_note is how
    that disclosure reaches the PERMANENT ledger row itself, not just an
    ephemeral run report. Default None must reproduce the plain template
    byte-for-byte (most callers pass nothing here)."""
    idx = pd.bdate_range("2020-01-02", periods=300)
    result = _hand_built_result(verdict="survived", idx=idx, n_sessions=300)
    ledger = str(tmp_path / "ledger.jsonl")

    record_abi1_outcome(
        ledger, "fam_extra_note", "run-1", "2026-09-09", result,
        catalog_entry=_catalog_entry(), instrument_set=["PFF"],
        cash_convention="sweep_cash_daily", benchmark="SPY",
        tax_policy="realization", ticker_label=None,
        extra_note="cost model: PUT=25.00bps (assumed_constant)",
        signal_source_path="scripts/run_abi1_mining.py",
        signal_source_sha256="a" * 64)

    row = _last_outcome(ledger)
    assert row["note"] == (
        "cash_convention=sweep_cash_daily; benchmark=SPY; "
        "tax_policy=realization; cost model: PUT=25.00bps (assumed_constant)")


def test_record_abi1_outcome_cut_dsr_routes_through_a_real_evaluate_call(tmp_path):
    """At least one verdict must be reached via
    a REAL account_relative.evaluate() call (the fixture recipe from
    test_mining_account_relative.py's own test_evaluate_cuts_on_dsr_before_
    risk), not a hand-built dict -- otherwise nothing in this suite would
    catch a future drift between record_abi1_outcome's assumed key paths
    and evaluate()'s real return shape."""
    idx = pd.bdate_range("2020-01-02", periods=400)
    strat = pd.Series(0.0, index=idx)
    mask = pd.Series(False, index=idx)
    strat.iloc[0:400] = 0.00005
    mask.iloc[0:400] = True
    result = account_relative.evaluate(
        strategy_daily=strat, in_market=mask,
        benchmark_daily=pd.Series(0.0004, index=idx),
        cash_daily=pd.Series(0.0001, index=idx),
        n_trials=100.0, sr_variance=0.0002, tax_policy="none")
    assert result["verdict"] == "cut_dsr"  # fixture sanity, not this test's subject

    ledger = str(tmp_path / "ledger.jsonl")
    stage = record_abi1_outcome(
        ledger, "fam_cut_dsr", "run-1", "2026-09-09", result,
        catalog_entry=_catalog_entry(), instrument_set=["QQQ"],
        cash_convention="sweep_cash_daily", benchmark="SPY",
        tax_policy="none", signal_source_path="scripts/run_abi1_mining.py",
        signal_source_sha256="b" * 64)

    assert stage == "cut_dsr"
    row = _last_outcome(ledger)
    assert row["stage"] == "cut_dsr"
    assert row["construction"] == "ABI-1"
    assert row["dsr_value"] == pytest.approx(result["layer1"]["dsr_value"])
    assert row["sharpe_in_sample"] == pytest.approx(
        result["layer1"]["sharpe_daily_excess"])
    assert row["evidence"] is None
    assert row["ticker_label"] is None
    assert row["note"] == ("cash_convention=sweep_cash_daily; "
                           "benchmark=SPY; tax_policy=none")


def test_record_abi1_outcome_cut_holdout_writes_a_scored_row(tmp_path):
    idx = pd.bdate_range("2015-01-02", periods=1200)
    result = _hand_built_result(
        verdict="cut_holdout", idx=idx, n_sessions=444,
        holdout_split_applied=True, holdout_sessions=756,
        in_sample_sessions=444, n_holdout=756, holdout_refusal=None)
    ledger = str(tmp_path / "ledger.jsonl")

    stage = record_abi1_outcome(
        ledger, "fam_cut_holdout", "run-1", "2026-09-09", result,
        catalog_entry=_catalog_entry(), instrument_set=["PFF"],
        cash_convention="sweep_cash_daily", benchmark="SPY",
        tax_policy="realization", signal_source_path="x",
        signal_source_sha256="c" * 64)

    assert stage == "cut_holdout"
    row = _last_outcome(ledger)
    assert row["stage"] == "cut_holdout"
    assert row["construction"] == "ABI-1"
    assert row["dsr_value"] == pytest.approx(0.99)
    assert row["evidence"] is None


def test_record_abi1_outcome_cut_risk_refusal_writes_a_scored_row(tmp_path):
    idx = pd.bdate_range("2015-01-02", periods=800)
    result = _hand_built_result(
        verdict="cut_risk_refusal", idx=idx, n_sessions=800,
        in_sample_sessions=800)
    ledger = str(tmp_path / "ledger.jsonl")

    stage = record_abi1_outcome(
        ledger, "fam_cut_risk", "run-1", "2026-09-09", result,
        catalog_entry=_catalog_entry(), instrument_set=["PFF"],
        cash_convention="sweep_cash_daily", benchmark="SPY",
        tax_policy="none", signal_source_path="x",
        signal_source_sha256="d" * 64)

    assert stage == "cut_risk_refusal"
    row = _last_outcome(ledger)
    assert row["stage"] == "cut_risk_refusal"
    assert row["construction"] == "ABI-1"
    assert row["dsr_value"] == pytest.approx(0.99)
    assert row["evidence"] is None


def test_record_abi1_outcome_not_scoreable_writes_the_real_evidence_payload(tmp_path):
    idx = pd.bdate_range("2020-01-02", periods=40)
    result = _hand_built_result(
        verdict="not_scoreable", idx=idx, n_sessions=40, dsr_value=None,
        sharpe_daily_excess=None, in_sample_sessions=40,
        holdout_refusal="in_sample_too_short",
        reason="only 40 account sessions, below MIN_ACCOUNT_SESSIONS (252)")
    ledger = str(tmp_path / "ledger.jsonl")

    stage = record_abi1_outcome(
        ledger, "fam_not_scoreable", "run-1", "2026-09-09", result,
        catalog_entry=_catalog_entry(), instrument_set=["PFF"],
        cash_convention="sweep_cash_daily", benchmark="SPY",
        tax_policy="realization", signal_source_path="x",
        signal_source_sha256="e" * 64)

    assert stage == "insufficient_evidence"
    row = _last_outcome(ledger)
    assert row["stage"] == "insufficient_evidence"
    assert row["construction"] == "ABI-1"
    assert row["dsr_value"] is None
    assert row["sharpe_in_sample"] is None
    payload = row["evidence"]
    assert payload["reason_code"] == "abi1_not_scoreable"
    assert payload["observed"]["n_sessions"] == 40
    assert payload["observed"]["account_start"] == idx[0].strftime("%Y-%m-%d")
    assert payload["observed"]["account_end"] == idx[-1].strftime("%Y-%m-%d")
    assert payload["observed"]["cash_convention"] == "sweep_cash_daily"
    assert payload["observed"]["benchmark"] == "SPY"
    # the REAL payload this function assembles, validated at write time --
    # not a hand-built fixture.
    evidence.validate(payload, stage="insufficient_evidence", at_write_time=True)


def test_record_abi1_outcome_holdout_not_evaluable_pins_the_real_tail_window(tmp_path):
    """pinned_window must be the REAL tail slice of
    result["account_daily"].index, not None -- verified against a
    hand-computed expectation from the same index."""
    idx = pd.bdate_range("2015-01-02", periods=1200)
    result = _hand_built_result(
        verdict="holdout_not_evaluable", idx=idx, n_sessions=444,
        holdout_split_applied=True, holdout_sessions=756,
        in_sample_sessions=444, n_holdout=756,
        holdout_refusal="zero_variance")
    ledger = str(tmp_path / "ledger.jsonl")

    stage = record_abi1_outcome(
        ledger, "fam_holdout_ne", "run-1", "2026-09-09", result,
        catalog_entry=_catalog_entry(), instrument_set=["PFF"],
        cash_convention="sweep_cash_daily", benchmark="SPY",
        tax_policy="realization", signal_source_path="x",
        signal_source_sha256="f" * 64)

    assert stage == "insufficient_evidence"
    row = _last_outcome(ledger)
    assert row["stage"] == "insufficient_evidence"
    assert row["construction"] == "ABI-1"
    assert row["dsr_value"] is None
    payload = row["evidence"]
    assert payload["reason_code"] == "abi1_holdout_not_evaluable"
    assert payload["observed"]["in_sample_dsr_value"] == pytest.approx(0.99)
    assert payload["observed"]["n_holdout_sessions"] == 756
    assert payload["observed"]["holdout_refusal"] == "zero_variance"
    expected_start = idx[-756].strftime("%Y-%m-%d")
    expected_end = idx[-1].strftime("%Y-%m-%d")
    assert payload["holdout"]["holdout_start"] == expected_start
    assert payload["holdout"]["holdout_end"] == expected_end
    # account_start/account_end stay the FULL span, separate from the
    # pinned holdout tail derived above.
    assert payload["observed"]["account_start"] == idx[0].strftime("%Y-%m-%d")
    assert payload["observed"]["account_end"] == idx[-1].strftime("%Y-%m-%d")
    evidence.validate(payload, stage="insufficient_evidence", at_write_time=True)


def test_record_abi1_outcome_rejects_a_non_positive_holdout_sessions_tail(tmp_path):
    """A hand-built result with
    holdout_split_applied=True and holdout_sessions=0 must be rejected, not
    silently produce account_index[-0] == account_index[0] -- which would
    mislabel the family's entire account span as the holdout tail in an
    append-only ledger row."""
    idx = pd.bdate_range("2015-01-02", periods=1200)
    result = _hand_built_result(
        verdict="holdout_not_evaluable", idx=idx,
        holdout_split_applied=True, holdout_sessions=0,
        n_holdout=0, holdout_refusal="zero_variance")
    ledger = str(tmp_path / "ledger.jsonl")

    with pytest.raises(ValueError, match="holdout_sessions"):
        record_abi1_outcome(
            ledger, "fam_bad_tail", "run-1", "2026-09-09", result,
            catalog_entry=_catalog_entry(), instrument_set=["PFF"],
            cash_convention="sweep_cash_daily", benchmark="SPY",
            tax_policy="realization", signal_source_path="x",
            signal_source_sha256="f" * 64)
    assert trial_ledger.latest_outcome(ledger, "fam_bad_tail") is None


def test_record_abi1_outcome_holdout_not_evaluable_falls_back_to_the_full_span_when_the_split_itself_refused(tmp_path):
    """The other branch: when split_holdout itself refused
    (no holdout sub-window ever existed), pinned_window must fall back to
    the full account span rather than a slice that does not correspond to
    any real holdout window."""
    idx = pd.bdate_range("2020-01-02", periods=300)
    result = _hand_built_result(
        verdict="holdout_not_evaluable", idx=idx, n_sessions=300,
        holdout_split_applied=False, holdout_sessions=0,
        in_sample_sessions=300, n_holdout=0,
        holdout_refusal="in_sample_too_short")
    ledger = str(tmp_path / "ledger.jsonl")

    record_abi1_outcome(
        ledger, "fam_holdout_ne_2", "run-1", "2026-09-09", result,
        catalog_entry=_catalog_entry(), instrument_set=["PFF"],
        cash_convention="sweep_cash_daily", benchmark="SPY",
        tax_policy="realization", signal_source_path="x",
        signal_source_sha256="0" * 64)

    row = _last_outcome(ledger)
    payload = row["evidence"]
    expected_start = idx[0].strftime("%Y-%m-%d")
    expected_end = idx[-1].strftime("%Y-%m-%d")
    assert payload["holdout"]["holdout_start"] == expected_start
    assert payload["holdout"]["holdout_end"] == expected_end
    evidence.validate(payload, stage="insufficient_evidence", at_write_time=True)


def test_record_abi1_outcome_rejects_an_unknown_verdict(tmp_path):
    idx = pd.bdate_range("2020-01-02", periods=300)
    result = _hand_built_result(verdict="not_a_real_verdict", idx=idx)
    ledger = str(tmp_path / "ledger.jsonl")

    with pytest.raises(ValueError, match="unknown ABI-1 verdict"):
        record_abi1_outcome(
            ledger, "fam_bad_verdict", "run-1", "2026-09-09", result,
            catalog_entry=_catalog_entry(), instrument_set=["PFF"],
            cash_convention="sweep_cash_daily", benchmark="SPY",
            tax_policy="realization", signal_source_path="x",
            signal_source_sha256="1" * 64)


def test_record_abi1_registers_once_with_actual_scored_holdout(monkeypatch):
    """Accounting storage is in memory; no real ledger or lock is opened."""
    from contextlib import nullcontext
    rows = []
    monkeypatch.setattr(trial_ledger, "_read_lines", lambda path: rows.copy())
    monkeypatch.setattr(trial_ledger, "_LedgerLock", lambda path: nullcontext())
    monkeypatch.setattr(trial_ledger, "_append_unlocked", lambda path, row: rows.append(row))
    account_idx = pd.date_range("2020-01-01", periods=8)
    result = _hand_built_result(
        verdict="cut_dsr", idx=account_idx, holdout_split_applied=True,
        holdout_sessions=2, in_sample_sessions=2)
    # Scored calendar differs from account calendar: pin the scored tail.
    result["log_excess"] = pd.Series(0.01, index=pd.to_datetime(
        ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]))
    kwargs = dict(catalog_entry=_catalog_entry(), instrument_set=["PFF"],
                  cash_convention="sweep_cash_daily", benchmark="SPY",
                  tax_policy="none", signal_source_path="fixture.py",
                  signal_source_sha256="a" * 64)
    record_abi1_outcome("memory-only", "fixture", "run-a", "2026-09-09", result, **kwargs)
    assert trial_ledger.cumulative_family_count("memory-only") == 1
    # outcome is written BEFORE first_test: a
    # record_outcome failure must never leave an orphaned first_test pin
    # with no corresponding result -- see test_record_abi1_outcome_failure_
    # never_registers_first_test below for the failure-path proof.
    assert rows[0]["event"] == "outcome"
    assert rows[1] == dict(event="first_test", family_id="fixture",
                           tested_date="2026-09-09", holdout_start="2020-01-06",
                           holdout_end="2020-01-07", representative_returns=[], run_id="run-a")
    # A second attempt for the SAME family raises
    # instead of silently no-op-ing on the outcome side while staying
    # idempotent on the first_test side -- the chokepoint itself enforces
    # the one-shot invariant rather than trusting every caller's own
    # (optional, racy) already_scored prefilter.
    with pytest.raises(trial_ledger.OutcomeAlreadyRecorded):
        record_abi1_outcome("memory-only", "fixture", "run-b", "2026-09-17", result, **kwargs)
    assert trial_ledger.cumulative_family_count("memory-only") == 1
    assert len([r for r in rows if r["event"] == "first_test"]) == 1
    assert len([r for r in rows if r["event"] == "outcome"]) == 1
    assert trial_ledger.get_pinned_holdout_window("memory-only", "fixture")["tested_date"] == "2026-09-09"


def test_record_abi1_outcome_failure_never_registers_first_test(monkeypatch):
    """record_outcome_if_not_scored is
    the chokepoint that validates stage/evidence shape and can raise. If it
    raises, this family must NOT end up with a first_test pin and no
    outcome -- that phantom row would inflate cumulative_family_count for a
    family that was never actually scored. Proven by making
    record_outcome_if_not_scored raise and asserting record_first_test was
    never called."""
    rows = []
    monkeypatch.setattr(trial_ledger, "_read_lines", lambda path: rows.copy())

    def _boom(*args, **kwargs):
        raise ValueError("simulated record_outcome failure")

    monkeypatch.setattr(trial_ledger, "record_outcome_if_not_scored", _boom)
    first_test_calls = []
    monkeypatch.setattr(
        trial_ledger, "record_first_test",
        lambda *a, **k: first_test_calls.append((a, k)))

    account_idx = pd.date_range("2020-01-01", periods=8)
    result = _hand_built_result(
        verdict="cut_dsr", idx=account_idx, holdout_split_applied=True,
        holdout_sessions=2, in_sample_sessions=2)
    kwargs = dict(catalog_entry=_catalog_entry(), instrument_set=["PFF"],
                  cash_convention="sweep_cash_daily", benchmark="SPY",
                  tax_policy="none", signal_source_path="fixture.py",
                  signal_source_sha256="a" * 64)
    with pytest.raises(ValueError, match="simulated record_outcome failure"):
        record_abi1_outcome("memory-only", "fixture", "run-a", "2026-09-09",
                            result, **kwargs)
    assert first_test_calls == []


def test_rotation_flat_segments_ignore_sold_assets_post_switch_move():
    idx = pd.bdate_range("2020-01-01", periods=3)
    frames = {"A": _rotation_frame([100, 100, 100], [100, 100, 200], idx),
              "B": _rotation_frame([100] * 3, [100] * 3, idx)}
    curve = build_rotation_curve(frames, lambda data: "A" if len(data["A"]) == 1 else "B",
                                 cost_bps_by_asset={"A": 0.0, "B": 0.0})
    assert curve["trades_df"]["gross_ret"].tolist() == [0.0, 0.0]
    assert curve["strategy_daily"].tolist() == [0.0, 0.0, 0.0]


def test_rotation_switch_compounds_overnight_and_new_intraday_legs():
    idx = pd.bdate_range("2020-01-01", periods=3)
    frames = {"A": _rotation_frame([100, 100, 110], [100, 100, 110], idx),
              "B": _rotation_frame([100] * 3, [100, 100, 120], idx)}
    curve = build_rotation_curve(frames, lambda data: "A" if len(data["A"]) == 1 else "B",
                                 cost_bps_by_asset={"A": 0.0, "B": 0.0})
    assert curve["strategy_daily"].iloc[-1] == pytest.approx(0.32)
    assert (1 + curve["strategy_daily"]).prod() == pytest.approx(1.32)


def test_rotation_exit_to_cash_preserves_overnight_return_in_account():
    idx = pd.bdate_range("2020-01-01", periods=4)
    frames = {"A": _rotation_frame([100, 100, 110, 110], [100, 100, 200, 200], idx),
              "B": _rotation_frame([100] * 4, [100] * 4, idx)}
    curve = build_rotation_curve(frames, lambda data: "A" if len(data["A"]) == 1 else None,
                                 cost_bps_by_asset={"A": 0.0, "B": 0.0})
    account = account_relative.account_daily_returns(
        strategy_daily=curve["strategy_daily"], in_market=curve["in_market"],
        cash_daily=pd.Series(0.001, index=idx))
    assert account.iloc[2] == pytest.approx(0.10)
    assert account.iloc[3] == pytest.approx(0.001)


@pytest.mark.parametrize("closes, expected", [
    ([100, 110, 110], 0.10),
    ([100, 200, 100], 0.0),
])
def test_top_n_pool_dispatch_allocates_selected_capital_without_daily_rebalancing(closes, expected):
    from types import SimpleNamespace
    scripts = os.path.abspath(os.path.join(_SRC, "..", "scripts"))
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import run_abi1_mining
    symbols = [f"S{i}" for i in range(11)]
    idx = pd.bdate_range("2020-01-01", periods=3)
    frames = {s: _rotation_frame([100] * 3, closes if i < 3 else [100] * 3, idx)
              for i, s in enumerate(symbols)}
    # For the round-trip test, only one of the three selected assets moves.
    if expected == 0.0:
        for s in symbols[1:3]:
            frames[s] = _rotation_frame([100] * 3, [100] * 3, idx)
    module = SimpleNamespace(
        HOLD_TOP_N=3,
        make_signal_fn=lambda symbol, index, data: (
            lambda frame, i: SignalResult(entry=(symbol in symbols[:3] and i == 0), exit=(i == 2))))
    plan = dict(spec={"shape": "pooled_single_asset", "signal": []}, module=module,
                effective=symbols, costs={s: {"cost_bps": 0.0} for s in symbols})
    curve = run_abi1_mining._build_curve(plan, frames)
    assert (1 + curve["strategy_daily"]).prod() - 1 == pytest.approx(expected)
    if expected == 0.0:
        assert curve["strategy_daily"].iloc[1] == pytest.approx(1 / 3)
        assert curve["strategy_daily"].iloc[2] == pytest.approx(-0.25)


def test_top_n_pool_refuses_incomplete_selection():
    idx = pd.bdate_range("2020-01-01", periods=3)
    frames = {s: _rotation_frame([100] * 3, [100, 110, 110], idx) for s in ("A", "B", "C")}
    signals = {s: (lambda frame, i, symbol=s: SignalResult(
        entry=(symbol != "C" and i == 0), exit=(i == 2))) for s in frames}
    with pytest.raises(ValueError, match="complete synchronized cohorts"):
        build_pooled_curve(signals, frames, cost_bps_by_symbol={s: 0 for s in frames},
                           selected_count=3)


def test_record_abi1_no_holdout_registers_null_dates_and_preserves_prior_pin(monkeypatch):
    from contextlib import nullcontext
    rows = [dict(event="first_test", family_id="existing", tested_date="2026-08-01",
                 holdout_start="2023-01-01", holdout_end="2025-12-31",
                 representative_returns=[], run_id="original")]
    monkeypatch.setattr(trial_ledger, "_read_lines", lambda path: rows.copy())
    monkeypatch.setattr(trial_ledger, "_LedgerLock", lambda path: nullcontext())
    monkeypatch.setattr(trial_ledger, "_append_unlocked", lambda path, row: rows.append(row))
    result = _hand_built_result(verdict="cut_dsr", idx=pd.bdate_range("2020-01-01", periods=4))
    kwargs = dict(catalog_entry=_catalog_entry(), instrument_set=["PFF"],
                  cash_convention="sweep_cash_daily", benchmark="SPY", tax_policy="none",
                  signal_source_path="fixture.py", signal_source_sha256="a" * 64)
    for family in ("existing", "new"):
        record_abi1_outcome("memory-only", family, "run-a", "2026-09-09", result, **kwargs)
    assert trial_ledger.cumulative_family_count("memory-only") == 2
    assert trial_ledger.get_pinned_holdout_window("memory-only", "existing") == {
        "tested_date": "2026-08-01", "holdout_start": "2023-01-01", "holdout_end": "2025-12-31"}
    assert trial_ledger.get_pinned_holdout_window("memory-only", "new") == {
        "tested_date": "2026-09-09", "holdout_start": None, "holdout_end": None}


# ---- unscored_note: an opt-in note for the two UNSCORED rows (history-end) ----

_UNSCORED_NOTE = "HISTORY END (opt-in setting): fixture clause"


def _unscored_call(ledger, family_id, result, **extra):
    return record_abi1_outcome(
        ledger, family_id, "run-1", "2026-09-09", result,
        catalog_entry=_catalog_entry(), instrument_set=["PFF"],
        cash_convention="sweep_cash_daily", benchmark="SPY",
        tax_policy="realization", signal_source_path="x",
        signal_source_sha256="e" * 64, **extra)


def _not_scoreable_result():
    return _hand_built_result(
        verdict="not_scoreable", idx=pd.bdate_range("2020-01-02", periods=40),
        n_sessions=40, dsr_value=None, sharpe_daily_excess=None,
        in_sample_sessions=40, holdout_refusal="in_sample_too_short",
        reason="only 40 account sessions, below MIN_ACCOUNT_SESSIONS (252)")


def _holdout_not_evaluable_result():
    return _hand_built_result(
        verdict="holdout_not_evaluable", idx=pd.bdate_range("2015-01-02", periods=1200),
        n_sessions=444, holdout_split_applied=True, holdout_sessions=756,
        in_sample_sessions=444, n_holdout=756, holdout_refusal="zero_variance")


@pytest.mark.parametrize("make_result", [_not_scoreable_result,
                                         _holdout_not_evaluable_result])
def test_record_abi1_outcome_unscored_note_reaches_both_unscored_rows(tmp_path, make_result):
    """A run that ends unscored still spends a permanent ledger slot (the
    construction=ABI-1 re-run guard), so an opt-in setting such as a history
    end must be readable on THAT row too, not only on the four scored ones."""
    ledger = str(tmp_path / "ledger.jsonl")
    stage = _unscored_call(ledger, "fam_unscored_note", make_result(),
                           unscored_note=_UNSCORED_NOTE)
    assert stage == "insufficient_evidence"
    row = _last_outcome(ledger)
    assert row["stage"] == "insufficient_evidence"
    assert row["note"] == _UNSCORED_NOTE
    assert row["dsr_value"] is None
    evidence.validate(row["evidence"], stage="insufficient_evidence",
                      at_write_time=True)


@pytest.mark.parametrize("make_result", [_not_scoreable_result,
                                         _holdout_not_evaluable_result])
def test_record_abi1_outcome_unscored_rows_stay_noteless_unless_opted_in(tmp_path, make_result):
    """Default None leaves the unscored row's note empty, and extra_note
    alone (which many driver scripts pass) still does NOT reach an unscored
    row: only the opt-in unscored_note keyword changes anything."""
    ledger = str(tmp_path / "ledger.jsonl")
    _unscored_call(ledger, "fam_default", make_result())
    assert _last_outcome(ledger)["note"] is None
    ledger2 = str(tmp_path / "ledger2.jsonl")
    _unscored_call(ledger2, "fam_extra_only", make_result(),
                   extra_note="cost model: fixture disclosure")
    assert _last_outcome(ledger2)["note"] is None
    ledger3 = str(tmp_path / "ledger3.jsonl")
    _unscored_call(ledger3, "fam_empty", make_result(), unscored_note="")
    assert _last_outcome(ledger3)["note"] is None


def test_record_abi1_outcome_unscored_note_is_ignored_on_a_scored_row(tmp_path):
    """The scored template is untouched: it never reads unscored_note."""
    idx = pd.bdate_range("2020-01-02", periods=300)
    result = _hand_built_result(verdict="survived", idx=idx, n_sessions=300)
    ledger = str(tmp_path / "ledger.jsonl")
    _unscored_call(ledger, "fam_scored", result, unscored_note=_UNSCORED_NOTE,
                   extra_note="scored-only disclosure")
    assert _last_outcome(ledger)["note"] == (
        "cash_convention=sweep_cash_daily; benchmark=SPY; "
        "tax_policy=realization; scored-only disclosure")
