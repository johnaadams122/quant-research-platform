import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from mining import wheel_engine
from mining import options_positions
from options_engine import bs_price


def _price_path(*segments, start="2020-01-02"):
    """Concatenate flat/linear segments into one price Series. Each
    segment is (n_days, start_price, end_price)."""
    idx_all, vals_all = [], []
    cursor = pd.Timestamp(start)
    for n, p0, p1 in segments:
        idx = pd.bdate_range(cursor, periods=n)
        vals_all.extend(np.linspace(p0, p1, n))
        idx_all.extend(idx)
        cursor = idx[-1] + pd.Timedelta(days=1)
    return pd.Series(vals_all, index=pd.DatetimeIndex(idx_all))


def test_run_wheel_stays_csp_and_expires_otm_on_a_flat_market():
    # A perfectly flat, zero-noise price path makes a 15-delta OTM put's
    # BSM price decay perfectly monotonically toward 0 as T -> 0 (positive
    # theta, nothing to offset it), so it crosses ANY positive
    # profit-target threshold strictly before literal expiry on every
    # cycle -- verified empirically by sweeping profit_target_pct from 0.9
    # up to 0.9999 on the plain flat path: active_close fired every cycle,
    # never expiry_otm. Real "quiet" markets aren't literally
    # zero-volatility day to day, so this local price series adds a small,
    # deterministic (NOT np.random) day-to-day wobble on top of the flat
    # 100-level base: a repeating linear decline of 4.5 (price units) over
    # each ~22-trading-day cycle (dte=21 -> a full unassisted cycle spans
    # days 0..21 inclusive = 22 rows), snapping back to the 100 level at
    # the start of the next cycle. Drifting toward the strike late in each
    # cycle ticks the put's extrinsic value back up just enough to break
    # the perfectly-monotonic decay, without ever approaching the
    # ~94.66-level strike (15-delta, dte=21, iv=0.20) closely enough to
    # risk assignment -- the path's minimum is 95.50, roughly a 0.84-point
    # buffer above the strike. This amplitude was chosen empirically from
    # the middle of a wide safe window (roughly 3.9 to 5.3): below 3.9 the
    # wobble is too small to break the monotonic crossing (still all
    # active_close), at/above 5.4 it dips far enough to risk assignment --
    # 4.5 is not a knife-edge value.
    n = 66
    price = _price_path((n, 100.0, 100.0))
    period = 22
    amplitude = 4.5
    t = np.arange(n)
    wobble = -amplitude * (t % period) / (period - 1)
    price = price + wobble
    iv = pd.Series(0.20, index=price.index)
    r = pd.Series(0.04, index=price.index)
    result = wheel_engine.run_wheel(price, iv, r, div_yield=0.0, target_delta=0.15,
                                     dte=21, profit_target_pct=0.9,
                                     cost_pct_of_premium=0.02)
    reasons = [rl["reason"] for rl in result["realizations"]]
    assert reasons.count("expiry_otm") >= 2
    assert "assigned" not in reasons


def test_run_wheel_assigns_on_a_crash_then_calls_away_on_a_rally():
    # Cycle 1 (days 0-21): flat, expires OTM. Cycle 2 (days 22-43): a
    # sharp crash through the strike -> assignment. Cycle 3+ (days 44+):
    # a strong rally well above spot -> called away.
    price = _price_path((22, 100.0, 100.0), (22, 100.0, 60.0), (44, 60.0, 130.0))
    iv = pd.Series(0.25, index=price.index)
    r = pd.Series(0.04, index=price.index)
    result = wheel_engine.run_wheel(price, iv, r, div_yield=0.0, target_delta=0.15,
                                     dte=21, profit_target_pct=0.99,
                                     cost_pct_of_premium=0.02)
    reasons = [rl["reason"] for rl in result["realizations"]]
    assert "assigned" in reasons
    assert "called_away" in reasons
    assert reasons.index("called_away") > reasons.index("assigned")


def test_run_wheel_nav_daily_is_always_positive_and_finite():
    price = _price_path((22, 100.0, 100.0), (22, 100.0, 40.0), (44, 40.0, 150.0))
    iv = pd.Series(0.40, index=price.index)
    r = pd.Series(0.04, index=price.index)
    result = wheel_engine.run_wheel(price, iv, r, div_yield=0.0, target_delta=0.15,
                                     dte=21, profit_target_pct=0.5,
                                     cost_pct_of_premium=0.02)
    nav = result["nav_daily"]
    assert len(nav) > 0
    assert np.isfinite(nav.to_numpy()).all()
    assert (nav > 0).all()


def test_run_wheel_in_market_is_always_true():
    price = _price_path((44, 100.0, 100.0))
    iv = pd.Series(0.20, index=price.index)
    r = pd.Series(0.04, index=price.index)
    result = wheel_engine.run_wheel(price, iv, r, div_yield=0.0, target_delta=0.15,
                                     dte=21, profit_target_pct=0.9,
                                     cost_pct_of_premium=0.02)
    assert result["in_market"].all()
    assert result["in_market"].index.equals(result["nav_daily"].index)


def test_run_wheel_active_close_fires_on_a_vol_crush():
    price = _price_path((22, 100.0, 101.0))
    iv = pd.Series(np.linspace(0.35, 0.05, 22), index=price.index)
    r = pd.Series(0.04, index=price.index)
    result = wheel_engine.run_wheel(price, iv, r, div_yield=0.0, target_delta=0.15,
                                     dte=21, profit_target_pct=0.5,
                                     cost_pct_of_premium=0.02)
    reasons = [rl["reason"] for rl in result["realizations"]]
    assert "active_close" in reasons


def test_run_wheel_realizations_have_required_fields():
    price = _price_path((22, 100.0, 100.0))
    iv = pd.Series(0.20, index=price.index)
    r = pd.Series(0.04, index=price.index)
    result = wheel_engine.run_wheel(price, iv, r, div_yield=0.0, target_delta=0.15,
                                     dte=21, profit_target_pct=0.9,
                                     cost_pct_of_premium=0.02)
    for rl in result["realizations"]:
        assert set(rl) >= {"entry_date", "exit_date", "nav_entry", "nav_exit", "reason"}
        assert rl["nav_entry"] > 0 and rl["nav_exit"] > 0
        assert rl["exit_date"] >= rl["entry_date"]


def test_run_wheel_truncates_cleanly_at_end_of_data():
    price = _price_path((10, 100.0, 100.0))  # shorter than one full dte=21 cycle
    iv = pd.Series(0.20, index=price.index)
    r = pd.Series(0.04, index=price.index)
    result = wheel_engine.run_wheel(price, iv, r, div_yield=0.0, target_delta=0.15,
                                     dte=21, profit_target_pct=0.9,
                                     cost_pct_of_premium=0.02)
    assert result["realizations"][-1]["reason"] == "end_of_data"
    assert len(result["nav_daily"]) == 10


def test_run_wheel_returns_empty_on_all_nan_iv():
    price = _price_path((10, 100.0, 100.0))
    iv = pd.Series(np.nan, index=price.index)
    r = pd.Series(0.04, index=price.index)
    result = wheel_engine.run_wheel(price, iv, r, div_yield=0.0, target_delta=0.15,
                                     dte=21, profit_target_pct=0.9,
                                     cost_pct_of_premium=0.02)
    assert len(result["nav_daily"]) == 0
    assert result["realizations"] == []


# ── Connecting-day mark-to-market regression tests ────────────────────────
#
# Regression tests: the connecting day must carry the real price move.
# `_advance_cycle` sets `cycle_base = path["nav"].iloc[0]` to the NEW cycle's
# own entry-day NAV, so a new cycle's first-day return ratio is 1.0 relative
# to itself. That is correct when the account held cash between cycles (a
# fresh CSP has no market exposure overnight), but wrong whenever the account
# continues holding a physical position (shares, or PMCC's LEAPS) across the
# cycle boundary: the overnight/connecting-day price move would be dropped,
# understating (or overstating) the account's return. These tests guard
# against that on a multi-day, multi-cycle stretch where shares or LEAPS are
# held across the boundary.
#
# Each test below inserts a deliberate, large one-day price jump exactly on
# the connecting day and pins the EXACT expected ratio (not just "changed"),
# so a connecting-day ratio of 1.0 fails.


def test_run_wheel_assigned_transition_attributes_the_connecting_day_return():
    # Cycle 1 (flat @100, 22 days): CSP expires OTM. Cycle 2 (crash
    # 100->60, 22 days): CSP assigned -- exits 2020-02-27, the very next
    # covered_call cycle enters 2020-02-28 holding the newly-acquired
    # shares. A deliberate -25% jump lands exactly on that entry day (the
    # connecting day), affecting nothing about cycle 1/2's own outcomes
    # (fully determined by data through 2020-02-27).
    price = _price_path((22, 100.0, 100.0), (22, 100.0, 60.0), (44, 60.0, 130.0))
    jump_date = pd.Timestamp("2020-02-28")
    price = price.copy()
    price.loc[jump_date:] = price.loc[jump_date:] * 0.75
    iv = pd.Series(0.25, index=price.index)
    r = pd.Series(0.04, index=price.index)
    kwargs = dict(div_yield=0.0, target_delta=0.15, dte=21, profit_target_pct=0.99,
                  cost_pct_of_premium=0.02)
    result = wheel_engine.run_wheel(price, iv, r, **kwargs)
    reasons = [rl["reason"] for rl in result["realizations"]]
    assert reasons[:2] == ["active_close", "assigned"]
    exit_date = pd.Timestamp("2020-02-27")
    entry_date = jump_date
    nav = result["nav_daily"]
    expected_ratio = float(price.loc[entry_date]) / float(price.loc[exit_date])
    actual_ratio = nav.loc[entry_date] / nav.loc[exit_date]
    assert actual_ratio == pytest.approx(expected_ratio, rel=1e-12)
    # Sanity: dropping the connecting-day move would give exactly 1.0 here
    # regardless of the jump -- assert we are nowhere near that.
    assert abs(actual_ratio - 1.0) > 0.1


def test_run_wheel_covered_call_rollover_attributes_the_connecting_day_return():
    # Cycle 1: flat @100, CSP expires OTM. Cycle 2: crash 100->60, CSP
    # assigned. Cycle 3+: flat @60 for a long stretch -- the covered call
    # keeps expiring/active-closing WITHOUT being called away (price never
    # rallies), so the SAME shares roll from one covered_call cycle into
    # the next covered_call cycle. First such rollover: exit 2020-03-16
    # (active_close) -> next entry 2020-03-17, still covered_call. A
    # deliberate -12% jump lands exactly on that connecting day.
    price = _price_path((22, 100.0, 100.0), (22, 100.0, 60.0), (90, 60.0, 60.0))
    jump_date = pd.Timestamp("2020-03-17")
    price = price.copy()
    price.loc[jump_date:] = price.loc[jump_date:] * 0.88
    iv = pd.Series(0.25, index=price.index)
    r = pd.Series(0.04, index=price.index)
    kwargs = dict(div_yield=0.0, target_delta=0.15, dte=21, profit_target_pct=0.99,
                  cost_pct_of_premium=0.02)
    result = wheel_engine.run_wheel(price, iv, r, **kwargs)
    reasons = [rl["reason"] for rl in result["realizations"]]
    assert reasons[:3] == ["active_close", "assigned", "active_close"]
    exit_date = pd.Timestamp("2020-03-16")
    entry_date = jump_date
    nav = result["nav_daily"]
    expected_ratio = float(price.loc[entry_date]) / float(price.loc[exit_date])
    actual_ratio = nav.loc[entry_date] / nav.loc[exit_date]
    assert actual_ratio == pytest.approx(expected_ratio, rel=1e-12)
    assert actual_ratio == pytest.approx(0.88, rel=1e-9)
    assert abs(actual_ratio - 1.0) > 0.1


def test_run_wheel_called_away_transition_is_not_bridged():
    # The shares are SOLD at the call strike exactly on a called_away exit
    # -- the account converts to cash that same day, so unlike "assigned"
    # or a covered_call rollover, there is nothing held overnight into the
    # next (fresh CSP) cycle. A deliberate -30% jump lands exactly on the
    # day after the called_away exit and must NOT be bridged (ratio stays
    # exactly 1.0, the correct cash-to-cash behavior).
    price = _price_path((22, 100.0, 100.0), (22, 100.0, 60.0), (44, 60.0, 130.0))
    iv = pd.Series(0.25, index=price.index)
    r = pd.Series(0.04, index=price.index)
    kwargs = dict(div_yield=0.0, target_delta=0.15, dte=21, profit_target_pct=0.99,
                  cost_pct_of_premium=0.02)
    baseline = wheel_engine.run_wheel(price, iv, r, **kwargs)
    called_away = next(rl for rl in baseline["realizations"] if rl["reason"] == "called_away")
    exit_date = called_away["exit_date"]
    entry_idx = list(price.index).index(exit_date) + 1
    entry_date = price.index[entry_idx]

    price2 = price.copy()
    price2.loc[entry_date:] = price2.loc[entry_date:] * 0.7
    result = wheel_engine.run_wheel(price2, iv, r, **kwargs)
    reasons = [rl["reason"] for rl in result["realizations"]]
    assert "called_away" in reasons  # premise: the jump didn't erase the transition
    nav = result["nav_daily"]
    assert nav.loc[entry_date] / nav.loc[exit_date] == pytest.approx(1.0, abs=1e-12)


def test_run_wheel_csp_to_csp_transition_is_not_bridged():
    # A CSP->CSP boundary (active_close/expiry_otm while never assigned)
    # holds cash the whole time -- no shares ever exist to carry overnight.
    # A deliberate large jump right after the first cycle's exit must NOT
    # be bridged.
    n = 66
    price = _price_path((n, 100.0, 100.0))
    period = 22
    amplitude = 4.5
    t = np.arange(n)
    wobble = -amplitude * (t % period) / (period - 1)
    price = price + wobble
    iv = pd.Series(0.20, index=price.index)
    r = pd.Series(0.04, index=price.index)
    kwargs = dict(div_yield=0.0, target_delta=0.15, dte=21, profit_target_pct=0.9,
                  cost_pct_of_premium=0.02)
    baseline = wheel_engine.run_wheel(price, iv, r, **kwargs)
    assert all(rl["reason"] != "assigned" for rl in baseline["realizations"])
    exit_date = baseline["realizations"][0]["exit_date"]
    entry_date = baseline["realizations"][1]["entry_date"]

    price2 = price.copy()
    price2.loc[entry_date:] = price2.loc[entry_date:] - 10.0
    result = wheel_engine.run_wheel(price2, iv, r, **kwargs)
    nav = result["nav_daily"]
    assert nav.loc[entry_date] / nav.loc[exit_date] == pytest.approx(1.0, abs=1e-12)


def test_run_pmcc_produces_realizations_and_positive_finite_nav():
    price = _price_path((300, 100.0, 110.0))  # a long, gently rising history
    iv = pd.Series(0.25, index=price.index)
    r = pd.Series(0.04, index=price.index)
    result = wheel_engine.run_pmcc(price, iv, r, div_yield=0.01, long_target_delta=0.80,
                                    long_dte=252, short_target_delta=0.25, short_dte=21,
                                    long_roll_dte_floor=63, roll_trigger_delta=0.60,
                                    profit_target_pct=0.5, cost_pct_of_premium=0.03,
                                    nav_floor_fraction=0.01)
    assert len(result["realizations"]) > 0
    nav = result["nav_daily"]
    assert len(nav) > 0
    assert np.isfinite(nav.to_numpy()).all()
    assert (nav > 0).all()


def test_run_pmcc_in_market_always_true():
    price = _price_path((150, 100.0, 100.0))
    iv = pd.Series(0.25, index=price.index)
    r = pd.Series(0.04, index=price.index)
    result = wheel_engine.run_pmcc(price, iv, r, div_yield=0.0, long_target_delta=0.80,
                                    long_dte=252, short_target_delta=0.25, short_dte=21,
                                    long_roll_dte_floor=63, roll_trigger_delta=0.60,
                                    profit_target_pct=0.5, cost_pct_of_premium=0.03,
                                    nav_floor_fraction=0.01)
    assert result["in_market"].all()
    assert result["in_market"].index.equals(result["nav_daily"].index)


def test_run_pmcc_short_leg_rolls_defensively_on_a_rally():
    price = _price_path((21, 100.0, 130.0))  # sharp rally deep-ITMs the short call fast
    iv = pd.Series(0.25, index=price.index)
    r = pd.Series(0.04, index=price.index)
    result = wheel_engine.run_pmcc(price, iv, r, div_yield=0.0, long_target_delta=0.80,
                                    long_dte=252, short_target_delta=0.25, short_dte=21,
                                    long_roll_dte_floor=63, roll_trigger_delta=0.60,
                                    profit_target_pct=0.99, cost_pct_of_premium=0.03,
                                    nav_floor_fraction=0.01)
    reasons = [rl["reason"] for rl in result["realizations"]]
    assert "active_close" in reasons


def test_run_pmcc_stays_positive_and_finite_over_a_history_long_enough_to_force_a_leaps_roll():
    # 252 (long_dte) - 63 (long_roll_dte_floor) = 189 trading days until a
    # roll is forced; 500 sessions guarantees at least one real roll.
    price = _price_path((250, 100.0, 90.0), (250, 90.0, 115.0))
    iv = pd.Series(0.30, index=price.index)
    r = pd.Series(0.04, index=price.index)
    result = wheel_engine.run_pmcc(price, iv, r, div_yield=0.01, long_target_delta=0.80,
                                    long_dte=252, short_target_delta=0.25, short_dte=21,
                                    long_roll_dte_floor=63, roll_trigger_delta=0.60,
                                    profit_target_pct=0.5, cost_pct_of_premium=0.03,
                                    nav_floor_fraction=0.01)
    nav = result["nav_daily"]
    assert np.isfinite(nav.to_numpy()).all()
    assert (nav > 0).all()
    # The bound allows for early closes: profit_target_pct=0.5 on this
    # gently-trending, iv=0.30 path makes the sub-cycles close early via
    # active_close (roughly half of short_dte=21 each), the same
    # profit-target-driven early-close behavior documented for run_wheel
    # elsewhere in this file (e.g.
    # test_run_wheel_active_close_fires_on_a_vol_crush), so a
    # one-cycle-per-short_dte estimate would undercount.
    assert 20 <= len(result["realizations"]) <= 60
    assert result["leaps_roll_count"] >= 1


def test_run_pmcc_leaps_rolls_far_less_often_than_the_short_leg():
    # The LEAPS leg must roll far less often than the short leg over a
    # multi-year synthetic path.
    price = _price_path((250, 100.0, 95.0), (250, 95.0, 108.0))
    iv = pd.Series(0.25, index=price.index)
    r = pd.Series(0.04, index=price.index)
    result = wheel_engine.run_pmcc(price, iv, r, div_yield=0.01, long_target_delta=0.80,
                                    long_dte=252, short_target_delta=0.25, short_dte=21,
                                    long_roll_dte_floor=63, roll_trigger_delta=0.60,
                                    profit_target_pct=0.5, cost_pct_of_premium=0.03,
                                    nav_floor_fraction=0.01)
    assert result["leaps_roll_count"] >= 1
    # 189 trading days per LEAPS cycle vs ~21 per short cycle -- roughly a
    # 9:1 ratio; require at least a generous 5:1 margin, not an exact one.
    assert result["leaps_roll_count"] * 5 < len(result["realizations"])


def test_run_pmcc_rejects_long_roll_dte_floor_not_exceeding_short_dte():
    price = _price_path((30, 100.0, 100.0))
    iv = pd.Series(0.25, index=price.index)
    r = pd.Series(0.04, index=price.index)
    with pytest.raises(ValueError, match="long_roll_dte_floor"):
        wheel_engine.run_pmcc(price, iv, r, div_yield=0.0, long_target_delta=0.80,
                               long_dte=252, short_target_delta=0.25, short_dte=21,
                               long_roll_dte_floor=21, roll_trigger_delta=0.60,
                               profit_target_pct=0.5, cost_pct_of_premium=0.03,
                               nav_floor_fraction=0.01)


def test_run_pmcc_rejects_long_reset_delta_not_below_long_target_delta():
    # A freshly entered/rolled/reset LEAPS' own delta is solved to land
    # AT long_target_delta (build_pmcc_leaps -> solve_strike_for_delta), so
    # if long_reset_delta were >= long_target_delta, the value-based reset
    # would silently fire on every single sub-cycle immediately after entry.
    # run_pmcc rejects that combination, the same cross-parameter check as
    # long_roll_dte_floor<=short_dte above.
    price = _price_path((30, 100.0, 100.0))
    iv = pd.Series(0.25, index=price.index)
    r = pd.Series(0.04, index=price.index)
    with pytest.raises(ValueError, match="long_reset_delta"):
        wheel_engine.run_pmcc(price, iv, r, div_yield=0.0, long_target_delta=0.80,
                               long_dte=252, short_target_delta=0.25, short_dte=21,
                               long_roll_dte_floor=63, roll_trigger_delta=0.60,
                               profit_target_pct=0.5, cost_pct_of_premium=0.03,
                               nav_floor_fraction=0.01, long_reset_delta=0.80)


def test_run_pmcc_labels_a_real_itm_at_true_expiry_correctly():
    # A non-active-closed exit is labelled by its real outcome: an
    # ITM-at-expiry short call is "short_assigned_at_expiry", never
    # "short_expiry_otm", including the case the roll_trigger_delta defense
    # cannot catch (a move confined entirely to a sub-cycle's LAST trading
    # day, where T_short_t=0 and close_signal can no longer fire by
    # construction).
    #
    # Fixture note (hand-verified): a PERFECTLY flat price makes the OTM
    # short call's own BSM price decay monotonically toward 0 as T -> 0, by definition
    # of "out of the money at expiry" -- so it crosses ANY high
    # profit_target_pct threshold (here 0.99) 1-2 sessions before literal
    # expiry regardless of the price never moving, pre-empting this very
    # scenario. This is the SAME phenomenon already documented in this
    # file's test_run_wheel_stays_csp_and_expires_otm_on_a_flat_market
    # ("crosses ANY positive profit-target threshold strictly before
    # literal expiry ... on the plain flat path"), just for a call instead
    # of a put. Verified directly against options_positions.pmcc_nav_path
    # in isolation: a literal flat-100 path trips close_signal at day 19
    # of 21 (short_price 0.0043 <= threshold 0.0084), two sessions before
    # the jump. A gentle 21-day ramp from 100 to 102 (still safely OTM of
    # the ~104.49 strike; short_delta stays <=0.25, far below
    # roll_trigger_delta=0.99) keeps the short call's extrinsic value
    # elevated enough that neither defense fires before the true-expiry
    # day -- verified: close_signal is False on every one of the 21
    # pre-expiry sessions, last pre-expiry short_price 0.0143 vs threshold
    # 0.0084 (real margin, not a knife-edge tuning). The sharp jump to 140
    # is still confined entirely to the true-expiry (22nd) day.
    n = 22
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    prices = np.concatenate([np.linspace(100.0, 102.0, 21), [140.0]])
    price = pd.Series(prices, index=idx)
    iv = pd.Series(0.20, index=idx)
    r = pd.Series(0.04, index=idx)
    result = wheel_engine.run_pmcc(price, iv, r, div_yield=0.0, long_target_delta=0.80,
                                    long_dte=252, short_target_delta=0.25, short_dte=21,
                                    long_roll_dte_floor=63, roll_trigger_delta=0.99,
                                    profit_target_pct=0.99, cost_pct_of_premium=0.03,
                                    nav_floor_fraction=0.01)
    reasons = [rl["reason"] for rl in result["realizations"]]
    assert "short_assigned_at_expiry" in reasons
    assert "short_expiry_otm" not in reasons  # the only cycle in this run really was ITM


def test_run_pmcc_realizations_have_required_fields():
    price = _price_path((60, 100.0, 100.0))
    iv = pd.Series(0.25, index=price.index)
    r = pd.Series(0.04, index=price.index)
    result = wheel_engine.run_pmcc(price, iv, r, div_yield=0.0, long_target_delta=0.80,
                                    long_dte=252, short_target_delta=0.25, short_dte=21,
                                    long_roll_dte_floor=63, roll_trigger_delta=0.60,
                                    profit_target_pct=0.5, cost_pct_of_premium=0.03,
                                    nav_floor_fraction=0.01)
    for rl in result["realizations"]:
        assert set(rl) >= {"entry_date", "exit_date", "nav_entry", "nav_exit", "reason"}
        assert rl["nav_entry"] > 0 and rl["nav_exit"] > 0


def test_run_pmcc_returns_empty_on_all_nan_iv():
    price = _price_path((10, 100.0, 100.0))
    iv = pd.Series(np.nan, index=price.index)
    r = pd.Series(0.04, index=price.index)
    result = wheel_engine.run_pmcc(price, iv, r, div_yield=0.0, long_target_delta=0.80,
                                    long_dte=252, short_target_delta=0.25, short_dte=21,
                                    long_roll_dte_floor=63, roll_trigger_delta=0.60,
                                    profit_target_pct=0.5, cost_pct_of_premium=0.03,
                                    nav_floor_fraction=0.01)
    assert len(result["nav_daily"]) == 0


def test_run_pmcc_stays_positive_through_a_crash_to_zero():
    # Regression test: a crash to zero must keep run_pmcc's NAV positive.
    # This is the scenario test_build_options_curve_strategy_daily_
    # never_below_negative_one[100.0-0.01-0.6] exercises, pinned directly at
    # the run_pmcc/wheel_engine level (not just the pmcc_nav_path unit level,
    # since that is the level the property test exercises and the
    # level a future regression could reappear at). Mirrors
    # test_run_put_ratio_spread_stays_positive_through_a_crash_to_zero's
    # pattern below, but needs the LONGER 130-day path (not that test's
    # 22 days) to reach the LEAPS-side degeneracy: with long_dte=252 and
    # long_roll_dte_floor=63, the LEAPS does not roll until ~189 days in,
    # so a shorter crash never puts run_pmcc through this scenario --
    # this exact 130-day/iv=0.60 path would raise ZeroDivisionError without the
    # pmcc_nav_path floor (a 0.0 floor dividing realized_nav_exit in
    # _advance_cycle), and could produce arbitrarily large negative NAVs with that
    # floor alone but no _advance_cycle guard (active_cost, computed from a
    # real short_price premium, exceeding an already near-floor "nav").
    n = 130
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    price = pd.Series(np.linspace(100.0, 0.01, n), index=idx)
    iv = pd.Series(0.60, index=idx)
    r = pd.Series(0.04, index=idx)
    result = wheel_engine.run_pmcc(price, iv, r, div_yield=0.0, long_target_delta=0.80,
                                    long_dte=252, short_target_delta=0.25, short_dte=21,
                                    long_roll_dte_floor=63, roll_trigger_delta=0.60,
                                    profit_target_pct=0.5, cost_pct_of_premium=0.03,
                                    nav_floor_fraction=0.01)
    nav = result["nav_daily"]
    assert len(nav) > 0
    assert np.isfinite(nav.to_numpy()).all()
    assert (nav > 0).all()  # strictly positive, not just non-negative


def test_run_pmcc_connecting_day_bridges_leaps_remark_across_sub_cycle_boundary():
    # PMCC analogue of run_wheel's connecting-day bridge -- the LEAPS persists
    # across nearly every short-call sub-cycle boundary (the whole point of
    # PMCC), so the bridge applies far more often than at run_wheel's own
    # covered-call rollover. First sub-cycle exits
    # 2020-01-20 (active_close); the next sub-cycle enters 2020-01-21
    # against the SAME LEAPS (leaps_roll_count stays 0 over this short a
    # path -- long_dte=252 vs a 60-day history). A deliberate +15% jump
    # lands exactly on that connecting day.
    #
    # Unlike run_wheel's share-price-ratio bridge, the expected factor here
    # is the LEAPS' OWN bs_price re-mark ratio, not the raw underlying price
    # ratio -- independently reconstructed below via the exact same
    # build_pmcc_leaps/bs_price calls run_pmcc itself makes, to prove the
    # bridge uses the derivative's own re-mark (S, K, T, r, iv all matter, not
    # just S).
    price = _price_path((60, 100.0, 105.0))
    jump_date = pd.Timestamp("2020-01-21")
    price = price.copy()
    price.loc[jump_date:] = price.loc[jump_date:] * 1.15
    iv = pd.Series(0.25, index=price.index)
    r = pd.Series(0.04, index=price.index)
    kwargs = dict(div_yield=0.0, long_target_delta=0.80, long_dte=252, short_target_delta=0.25,
                  short_dte=21, long_roll_dte_floor=63, roll_trigger_delta=0.60,
                  profit_target_pct=0.5, cost_pct_of_premium=0.03, nav_floor_fraction=0.01)
    result = wheel_engine.run_pmcc(price, iv, r, **kwargs)
    reasons = [rl["reason"] for rl in result["realizations"][:2]]
    assert reasons == ["active_close", "active_close"]
    assert result["leaps_roll_count"] == 0  # premise: no roll, so this IS a bridge case

    exit_date = pd.Timestamp("2020-01-20")
    entry_date = jump_date

    # Independently reconstruct the expected bridge ratio: the LEAPS was
    # entered fresh at trading day 0 (2020-01-02), same call run_pmcc's own
    # _enter_leaps(0) makes.
    leaps = options_positions.build_pmcc_leaps(
        float(price.iloc[0]), float(iv.iloc[0]), 252, target_delta=0.80,
        r=float(r.iloc[0]), q=0.0)
    idx_list = list(price.index)
    exit_i, entry_i = idx_list.index(exit_date), idx_list.index(entry_date)
    T_at_exit = leaps["T_entry"] - (exit_i - 0) / 252.0
    T_at_entry = leaps["T_entry"] - (entry_i - 0) / 252.0
    long_price_at_exit = bs_price(float(price.loc[exit_date]), leaps["K"], T_at_exit,
                                   float(r.loc[exit_date]), float(iv.loc[exit_date]),
                                   "call", q=0.0)
    long_price_at_entry = bs_price(float(price.loc[entry_date]), leaps["K"], T_at_entry,
                                    float(r.loc[entry_date]), float(iv.loc[entry_date]),
                                    "call", q=0.0)
    expected_ratio = long_price_at_entry / long_price_at_exit

    nav = result["nav_daily"]
    actual_ratio = nav.loc[entry_date] / nav.loc[exit_date]
    assert actual_ratio == pytest.approx(expected_ratio, rel=1e-9)
    # Must NOT match the raw underlying price ratio -- proves the bridge uses
    # the LEAPS' own re-mark, not a naive copy of run_wheel's share-price
    # ratio approach.
    raw_price_ratio = float(price.loc[entry_date]) / float(price.loc[exit_date])
    assert abs(actual_ratio - raw_price_ratio) > 0.1
    # Sanity: dropping the connecting-day move would give exactly 1.0
    # regardless of the jump.
    assert abs(actual_ratio - 1.0) > 0.1


def test_run_pmcc_leaps_roll_boundary_is_not_bridged():
    # A LEAPS roll buys a genuinely fresh position -- the old LEAPS is not
    # carried forward, same shape as run_wheel's called_away exclusion, so
    # there must be no connecting-day bridge exactly at a roll boundary.
    # This path (the same fixture as
    # test_run_pmcc_stays_positive_and_finite_over_a_history_long_enough_to_force_a_leaps_roll)
    # forces 2 LEAPS rolls; the first
    # roll boundary lands at realization 21 -> 22 (exit 2020-10-05 -> entry
    # 2020-10-06). A deliberate +25% jump lands exactly on that entry day.
    price = _price_path((250, 100.0, 90.0), (250, 90.0, 115.0))
    iv = pd.Series(0.30, index=price.index)
    r = pd.Series(0.04, index=price.index)
    kwargs = dict(div_yield=0.01, long_target_delta=0.80, long_dte=252, short_target_delta=0.25,
                  short_dte=21, long_roll_dte_floor=63, roll_trigger_delta=0.60,
                  profit_target_pct=0.5, cost_pct_of_premium=0.03, nav_floor_fraction=0.01)
    baseline = wheel_engine.run_pmcc(price, iv, r, **kwargs)
    assert baseline["leaps_roll_count"] >= 1
    r21, r22 = baseline["realizations"][21], baseline["realizations"][22]
    exit_date, entry_date = r21["exit_date"], r22["entry_date"]

    price2 = price.copy()
    price2.loc[entry_date:] = price2.loc[entry_date:] * 1.25
    result = wheel_engine.run_pmcc(price2, iv, r, **kwargs)
    # premise: the jump did not change WHEN the roll happens.
    assert result["leaps_roll_count"] == baseline["leaps_roll_count"]
    nav = result["nav_daily"]
    assert nav.loc[entry_date] / nav.loc[exit_date] == pytest.approx(1.0, abs=1e-12)


def test_run_pmcc_resets_leaps_via_delta_trigger_before_dte_floor():
    # Guards the value-based LEAPS reset (long_reset_delta). With a purely
    # time-based roll trigger (long_roll_dte_floor), a LEAPS crushed toward
    # worthless by a fast crash would keep being held and financed with new
    # short calls for months before its DTE alone forced a roll -- producing
    # a near-zero/negative entry_net_debit and degenerate-basis returns.
    #
    # This fixture (100 -> 20 over 60 trading days, iv=0.40) is hand-verified
    # directly against options_engine.bs_delta on the SAME K
    # build_pmcc_leaps would solve at entry: LEAPS delta crosses below 0.40
    # between day 25 (0.4071) and day 26 (0.3853), well inside the
    # long_roll_dte_floor=63 window (252-63=189 days before a time-based
    # roll would fire) -- long_roll_dte_floor is set to 189 (i.e. no
    # time-based roll possible at all within this 60-day path) specifically
    # to isolate the value-based trigger from the time-based one. With the
    # value trigger effectively disabled (a long_reset_delta the LEAPS delta
    # never crosses), sub-cycle entries land at the same trading days
    # 0,4,8,12,16,19,22,25,28,... and nav_entry (cycle_base) keeps shrinking
    # cycle after cycle as the crushed LEAPS is refinanced; the first entry
    # at or past day 26 is day 28 (cycle index 8), where the reset fires.
    n = 60
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    price = pd.Series(np.linspace(100.0, 20.0, n), index=idx)
    iv = pd.Series(0.40, index=idx)
    r = pd.Series(0.04, index=idx)
    result = wheel_engine.run_pmcc(price, iv, r, div_yield=0.0, long_target_delta=0.80,
                                    long_dte=252, short_target_delta=0.25, short_dte=21,
                                    long_roll_dte_floor=189, roll_trigger_delta=0.60,
                                    profit_target_pct=0.5, cost_pct_of_premium=0.03,
                                    nav_floor_fraction=0.01, long_reset_delta=0.40)
    assert result["leaps_roll_count"] == 0  # premise: isolates the value trigger
    assert result["leaps_value_reset_count"] >= 1
    nav = result["nav_daily"]
    assert np.isfinite(nav.to_numpy()).all()
    assert (nav > 0).all()


def test_run_pmcc_leaps_reset_reprices_with_fresh_time_to_expiry():
    # Regression test: a value-based reset that buys a fresh LEAPS
    # (via _enter_leaps) but forgets to also reset T_long_remaining to the
    # NEW LEAPS' own T_entry would price that fresh LEAPS' `long_mark_now`
    # against the OLD, stale (much smaller) T left over from the LEAPS just
    # discarded -- mispricing the entire post-reset sub-cycle.
    #
    # Reuses test_run_pmcc_resets_leaps_via_delta_trigger_before_dte_floor's
    # own fixture and its hand-verified cycle schedule: cycles 0-7's entry
    # days (0,4,8,12,16,19,22,25) are the same with or without the reset
    # (LEAPS delta stays >= 0.40 through cycle 7's entry at day 25, delta
    # 0.4071), so cycle index 8 (entry day 28, delta 0.3413) is reliably the
    # FIRST reset regardless of the reset's own internals -- this anchor does
    # not depend on trusting the code under test.
    n = 60
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    price = pd.Series(np.linspace(100.0, 20.0, n), index=idx)
    iv = pd.Series(0.40, index=idx)
    r = pd.Series(0.04, index=idx)
    cost_pct = 0.03
    result = wheel_engine.run_pmcc(price, iv, r, div_yield=0.0, long_target_delta=0.80,
                                    long_dte=252, short_target_delta=0.25, short_dte=21,
                                    long_roll_dte_floor=189, roll_trigger_delta=0.60,
                                    profit_target_pct=0.5, cost_pct_of_premium=cost_pct,
                                    nav_floor_fraction=0.01, long_reset_delta=0.40)

    reset_i = 28
    reset_date = idx[reset_i]
    reset_cycle = next(rl for rl in result["realizations"] if rl["entry_date"] == reset_date)
    S0 = float(price.iloc[reset_i])
    iv0 = float(iv.iloc[reset_i])
    r0 = float(r.iloc[reset_i])

    # Independent reconstruction using the SAME pure constructors run_pmcc
    # itself calls -- if T_long_remaining were wrongly left stale, the
    # production long_mark_now would NOT match this fresh (T=long_dte/252)
    # computation.
    fresh_leaps = options_positions.build_pmcc_leaps(
        S0, iv0, 252, target_delta=0.80, r=r0, q=0.0)
    fresh_short = options_positions.build_pmcc_short_call(
        S0, iv0, 21, target_delta=0.25, r=r0, q=0.0)
    expected_long_cost_gross = fresh_leaps["entry_cost"]
    expected_entry_net_debit = (
        expected_long_cost_gross * (1.0 + cost_pct)
        - fresh_short["entry_credit"] * (1.0 - cost_pct))

    assert reset_cycle["nav_entry"] == pytest.approx(expected_entry_net_debit, rel=1e-6), (
        f"post-reset cycle_base {reset_cycle['nav_entry']!r} does not match "
        f"an independently-computed FRESH-T_long_remaining entry_net_debit "
        f"{expected_entry_net_debit!r} -- the reset likely priced the new "
        f"LEAPS against a stale time-to-expiry left over from the discarded "
        f"one")


def test_run_put_ratio_spread_produces_realizations_and_positive_finite_nav():
    price = _price_path((66, 100.0, 100.0))
    iv = pd.Series(0.25, index=price.index)
    r = pd.Series(0.04, index=price.index)
    result = wheel_engine.run_put_ratio_spread(
        price, iv, r, div_yield=0.0, long_delta=0.30, short_delta=0.15,
        short_multiple=2, dte=21, profit_target_pct=0.9, cost_pct_of_premium=0.02)
    assert len(result["realizations"]) >= 2
    nav = result["nav_daily"]
    assert len(nav) > 0
    assert np.isfinite(nav.to_numpy()).all()
    assert (nav > 0).all()


def test_run_put_ratio_spread_in_market_always_true():
    price = _price_path((44, 100.0, 100.0))
    iv = pd.Series(0.25, index=price.index)
    r = pd.Series(0.04, index=price.index)
    result = wheel_engine.run_put_ratio_spread(
        price, iv, r, div_yield=0.0, long_delta=0.30, short_delta=0.15,
        short_multiple=2, dte=21, profit_target_pct=0.9, cost_pct_of_premium=0.02)
    assert result["in_market"].all()
    assert result["in_market"].index.equals(result["nav_daily"].index)


def test_run_put_ratio_spread_active_close_fires_on_decay():
    # Fixture note (hand-verified): the "price up 100->101 + IV decay
    # 0.35->0.05" shape of run_wheel/run_pmcc's own vol-crush tests does
    # NOT fire an active close here. Checked directly against
    # options_positions.ratio_spread_nav_path on these exact parameters:
    # entry_credit_gross = -0.3867 (this family's own negative-entry-credit
    # property at long_delta=0.30/short_delta=0.15/short_multiple=2 -- see
    # ratio_spread_nav_path's own docstring), and on that up path the
    # liability's most favorable point across the whole 22-day path only
    # reaches -0.5388, short of the profit_target_pct=0.5 threshold of -0.58
    # (79% of the way there, never crossing). A 1x2 put ratio spread's true
    # max-profit zone is at S_T = K_short (~90.82 here), not near-flat/up
    # price action -- the short-vega "price up + vol crush" shape that
    # works for run_wheel's naked short put/call never enters that zone
    # for this 2-leg net-debit structure. So this fixture uses a price path
    # declining moderately TOWARD (not through) K_short instead of up:
    # liability drops to -0.5999... by day 3 of this path, past the -0.58
    # threshold, firing active_close on 7 of 8 realized cycles (and
    # nav_daily stays finite and positive throughout).
    price = _price_path((22, 100.0, 92.0))
    iv = pd.Series(np.linspace(0.35, 0.05, 22), index=price.index)
    r = pd.Series(0.04, index=price.index)
    result = wheel_engine.run_put_ratio_spread(
        price, iv, r, div_yield=0.0, long_delta=0.30, short_delta=0.15,
        short_multiple=2, dte=21, profit_target_pct=0.5, cost_pct_of_premium=0.02)
    reasons = [rl["reason"] for rl in result["realizations"]]
    assert "active_close" in reasons


def test_run_put_ratio_spread_active_close_cost_reduces_nav_not_increases_it():
    # Guards the sign of the active-close cost: _advance_cycle charges
    # `cost_pct_of_premium * abs(float(close_row[exit_col]))`. Without the
    # abs() that would be fine for run_wheel/run_pmcc (their exit_col is
    # always a non-negative bs_price(...) output) but WRONG for
    # run_put_ratio_spread, whose exit_col is "liability" -- a SIGNED net
    # value (options_positions.ratio_spread_nav_path's
    # short_multiple*short_price - long_price) that is typically negative
    # at an active close for this pipeline's own catalog deltas. A
    # negative liability would make "active_cost" negative too, so
    # `nav - active_cost` would INCREASE nav instead of decreasing it -- a
    # transaction cost that paid the account. options_positions._cost_adjusted
    # applies the same sign rule to entry costs (see that helper's
    # docstring).
    #
    # This reuses the exact fixture from
    # test_run_put_ratio_spread_active_close_fires_on_decay (see that
    # test's own comment for why this price/iv shape reliably active-
    # closes on this family) -- verified directly against
    # options_positions.ratio_spread_nav_path in isolation: the very
    # first cycle active-closes at day 3 with liability=-0.5999... (well
    # into negative territory, where the sign matters).
    #
    # Same comparison style as test_options_positions.py's
    # test_ratio_spread_nav_path_cost_always_hurts_a_net_debit_position:
    # run twice at different cost_pct_of_premium and assert the
    # higher-cost run's realized NAV at the (same) active-close cycle is
    # LOWER, never higher. If the cost were taken on the signed liability
    # (no abs()), every one of these active-close cycles would flip the
    # wrong way (high-cost realized NAV > low-cost realized NAV).
    price = _price_path((22, 100.0, 92.0))
    iv = pd.Series(np.linspace(0.35, 0.05, 22), index=price.index)
    r = pd.Series(0.04, index=price.index)
    kwargs = dict(div_yield=0.0, long_delta=0.30, short_delta=0.15,
                  short_multiple=2, dte=21, profit_target_pct=0.5)
    low_cost = wheel_engine.run_put_ratio_spread(
        price, iv, r, cost_pct_of_premium=0.01, **kwargs)
    high_cost = wheel_engine.run_put_ratio_spread(
        price, iv, r, cost_pct_of_premium=0.05, **kwargs)

    low_reasons = [rl["reason"] for rl in low_cost["realizations"]]
    high_reasons = [rl["reason"] for rl in high_cost["realizations"]]
    assert "active_close" in low_reasons
    assert "active_close" in high_reasons
    # The exit-timing logic (close_signal) does not depend on
    # cost_pct_of_premium at all, so both runs must active-close on the
    # exact same cycles.
    assert low_reasons == high_reasons

    active_close_idxs = [n for n, rl in enumerate(low_reasons) if rl == "active_close"]
    assert len(active_close_idxs) >= 1
    for n in active_close_idxs:
        low_nav_exit = low_cost["realizations"][n]["nav_exit"]
        high_nav_exit = high_cost["realizations"][n]["nav_exit"]
        assert high_nav_exit < low_nav_exit, (
            f"cycle {n}: a higher transaction cost (5% vs 1% of premium) must "
            f"leave LESS realized NAV, not more -- got low_cost={low_nav_exit!r} "
            f"high_cost={high_nav_exit!r}")


def test_run_put_ratio_spread_stays_positive_through_a_crash_to_zero():
    n = 22
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    price = pd.Series(np.linspace(100.0, 0.01, n), index=idx)
    iv = pd.Series(0.60, index=idx)
    r = pd.Series(0.04, index=idx)
    result = wheel_engine.run_put_ratio_spread(
        price, iv, r, div_yield=0.0, long_delta=0.30, short_delta=0.15,
        short_multiple=2, dte=21, profit_target_pct=0.9, cost_pct_of_premium=0.02)
    assert (result["nav_daily"] > 0).all()


def test_run_put_ratio_spread_realizations_have_required_fields():
    price = _price_path((44, 100.0, 100.0))
    iv = pd.Series(0.25, index=price.index)
    r = pd.Series(0.04, index=price.index)
    result = wheel_engine.run_put_ratio_spread(
        price, iv, r, div_yield=0.0, long_delta=0.30, short_delta=0.15,
        short_multiple=2, dte=21, profit_target_pct=0.9, cost_pct_of_premium=0.02)
    for rl in result["realizations"]:
        assert set(rl) >= {"entry_date", "exit_date", "nav_entry", "nav_exit", "reason"}
        assert rl["nav_entry"] > 0 and rl["nav_exit"] > 0


def test_run_put_ratio_spread_truncates_at_end_of_data():
    price = _price_path((10, 100.0, 100.0))
    iv = pd.Series(0.20, index=price.index)
    r = pd.Series(0.04, index=price.index)
    result = wheel_engine.run_put_ratio_spread(
        price, iv, r, div_yield=0.0, long_delta=0.30, short_delta=0.15,
        short_multiple=2, dte=21, profit_target_pct=0.9, cost_pct_of_premium=0.02)
    assert result["realizations"][-1]["reason"] == "end_of_data"
    assert len(result["nav_daily"]) == 10


def test_run_put_ratio_spread_returns_empty_on_all_nan_iv():
    price = _price_path((10, 100.0, 100.0))
    iv = pd.Series(np.nan, index=price.index)
    r = pd.Series(0.04, index=price.index)
    result = wheel_engine.run_put_ratio_spread(
        price, iv, r, div_yield=0.0, long_delta=0.30, short_delta=0.15,
        short_multiple=2, dte=21, profit_target_pct=0.9, cost_pct_of_premium=0.02)
    assert len(result["nav_daily"]) == 0
