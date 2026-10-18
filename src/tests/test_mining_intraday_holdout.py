import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import intraday_holdout
from mining.intraday_backtest_engine import SignalResult, run_intraday_backtest


_ET = "America/New_York"


def _minute_frame(n_days: int, start_date: str = "2024-01-02", bars_per_day: int = 3):
    """A synthetic RTH-shaped minute-bar frame: `bars_per_day` bars/day (9:30,
    12:00, 15:59 ET) across `n_days` consecutive business days, tz-aware
    ("America/New_York") DatetimeIndex, matching intraday_data.fetch_intraday_bars's
    output shape. Open/Close are both strictly monotonically increasing bar
    over bar (so every day's mark-to-market leg is distinct and easy to
    reason about by hand) AND deliberately DISTINCT from each other within
    any one bar (Close = Open + a small tick) -- real market bars never
    have Open == Close exactly, and reconcile_holdout_window's own
    boundary re-pricing (synthetic entry = a bar's Open, real exit = a
    bar's Close) specifically depends on that being true even for the
    SAME bar (see that function's own docstring).
    """
    days = pd.bdate_range(start=start_date, periods=n_days)
    times = ["09:30", "12:00", "15:59"][:bars_per_day]
    idx = []
    for day in days:
        for t in times:
            idx.append(pd.Timestamp(f"{day.date()} {t}", tz=_ET))
    index = pd.DatetimeIndex(idx)
    n = len(index)
    opens = 100.0 + np.arange(n, dtype=float) * 0.10
    closes = opens + 0.02
    return pd.DataFrame({
        "Open": opens, "High": closes + 0.05, "Low": opens - 0.05,
        "Close": closes,
    }, index=index)


def _entry_exit_signal(entry_exec_i: int, exit_exec_i: int):
    """Enter so the position OPENS at absolute bar index `entry_exec_i`
    (next-bar-open execution, per run_intraday_backtest's own convention: the
    decision bar is entry_exec_i - 1), exit executes at bar `exit_exec_i`."""
    def signal_fn(frame, i):
        if i == entry_exec_i - 1:
            return SignalResult(entry=True, exit=False)
        if i == exit_exec_i:
            return SignalResult(entry=False, exit=True)
        return SignalResult(entry=False, exit=False)
    return signal_fn


# ---------------------------------------------------------------------------
# Low-level per-day leg helpers (_touched_day_reference_prices,
# _per_day_gross_legs, _apply_net_ret_plug) -- hand-computed, tiny examples,
# proving the formula itself before trusting reconcile_holdout_window's own
# wiring of it.
# ---------------------------------------------------------------------------

def test_per_day_gross_legs_long_telescopes_to_the_trade_total():
    from datetime import date
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    ref = {days[0]: 102.0, days[1]: 105.0, days[2]: 108.0}  # 108.0 == exit_px
    entry_px = 100.0
    legs = intraday_holdout._per_day_gross_legs(entry_px, days, ref, "long")
    assert legs[days[0]] == pytest.approx(0.02)
    assert legs[days[1]] == pytest.approx(105.0 / 102.0 - 1.0)
    assert legs[days[2]] == pytest.approx(108.0 / 105.0 - 1.0)
    telescoped = 1.0
    for d in days:
        telescoped *= (1.0 + legs[d])
    assert telescoped == pytest.approx(108.0 / 100.0)


def test_per_day_gross_legs_short_uses_fixed_entry_basis_and_sums_to_total():
    from datetime import date
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    ref = {days[0]: 102.0, days[1]: 105.0, days[2]: 108.0}
    entry_px = 100.0
    legs = intraday_holdout._per_day_gross_legs(entry_px, days, ref, "short")
    assert legs[days[0]] == pytest.approx((100.0 - 102.0) / 100.0)
    assert legs[days[1]] == pytest.approx((102.0 - 105.0) / 100.0)
    assert legs[days[2]] == pytest.approx((105.0 - 108.0) / 100.0)
    total = sum(legs.values())
    assert total == pytest.approx(1.0 - 108.0 / 100.0)  # matches short gross_ret


def test_apply_net_ret_plug_long_reconstructs_net_ret_exactly():
    from datetime import date
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    ref = {days[0]: 102.0, days[1]: 105.0, days[2]: 108.0}
    legs = intraday_holdout._per_day_gross_legs(100.0, days, ref, "long")
    plugged = intraday_holdout._apply_net_ret_plug(legs, days, "long", net_ret=0.075)
    assert plugged[days[2]] == pytest.approx(1.075 / (1.02 * (105.0 / 102.0)) - 1.0)
    total = 1.0
    for d in days:
        total *= (1.0 + plugged[d])
    assert total - 1.0 == pytest.approx(0.075)
    # Interior legs untouched by the plug.
    assert plugged[days[0]] == pytest.approx(legs[days[0]])
    assert plugged[days[1]] == pytest.approx(legs[days[1]])


def test_apply_net_ret_plug_short_reconstructs_net_ret_exactly():
    from datetime import date
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    ref = {days[0]: 102.0, days[1]: 105.0, days[2]: 108.0}
    legs = intraday_holdout._per_day_gross_legs(100.0, days, ref, "short")
    plugged = intraday_holdout._apply_net_ret_plug(legs, days, "short", net_ret=-0.09)
    interior_sum = legs[days[0]] + legs[days[1]]
    assert plugged[days[2]] == pytest.approx(-0.09 - interior_sum)
    assert sum(plugged.values()) == pytest.approx(-0.09)


def test_apply_net_ret_plug_single_day_reduces_to_net_ret_directly():
    from datetime import date
    days = [date(2024, 1, 2)]
    ref = {days[0]: 108.0}
    for direction in ("long", "short"):
        legs = intraday_holdout._per_day_gross_legs(100.0, days, ref, direction)
        plugged = intraday_holdout._apply_net_ret_plug(legs, days, direction, net_ret=0.05)
        assert plugged[days[0]] == pytest.approx(0.05)


def test_touched_day_reference_prices_uses_last_bar_close_and_exit_px():
    df = _minute_frame(n_days=3, bars_per_day=3)
    days = sorted(pd.unique(df.index.date))
    exit_px = 999.0  # deliberately NOT the real last-bar close, to prove it's used verbatim
    touched, ref = intraday_holdout._touched_day_reference_prices(
        df, days[0], days[2], exit_px)
    assert touched == list(days)
    assert ref[days[0]] == pytest.approx(
        float(df.loc[df.index.date == days[0]].iloc[-1]["Close"]))
    assert ref[days[1]] == pytest.approx(
        float(df.loc[df.index.date == days[1]].iloc[-1]["Close"]))
    assert ref[days[2]] == pytest.approx(exit_px)  # exit day uses exit_px, not its own Close


# ---------------------------------------------------------------------------
# reconcile_holdout_window
# ---------------------------------------------------------------------------

def test_position_opened_during_padding_and_held_into_holdout_gets_genuine_marks():
    """The central claim of the module docstring: a position opened during
    the padding period and held into the real holdout window must, once
    sliced, show genuine per-day mark-to-market values for every real
    holdout day it was held through -- NOT zeros followed by one lump sum on
    the exit day."""
    df = _minute_frame(n_days=10)  # days 0..9, 3 bars/day, 30 bars total
    days = sorted(pd.unique(df.index.date))
    assert len(days) == 10
    holdout_start = days[4].isoformat()  # day index 4
    holdout_end = days[9].isoformat()    # day index 9 (last day)

    # Position opens on day 1 (bar index 3, the first bar of day index 1;
    # decision bar is entry_exec_i-1=2, the last bar of day 0) -- squarely
    # inside the padding period (days 0-3). Exits on day 6 (bar index 18,
    # the first bar of day 6) -- inside the real holdout window.
    entry_exec_i = 3   # day index 1, bar 0
    exit_exec_i = 18   # day index 6, bar 0
    signal_fn = _entry_exit_signal(entry_exec_i, exit_exec_i)

    trades_df, daily_returns = run_intraday_backtest(
        df, signal_fn, cost_bps=0.0, tax_haircut_rate=0.0,
        max_hold_days=20, direction="long")

    assert len(trades_df) == 1
    trade = trades_df.iloc[0]
    assert trade["exit_datetime"].date() == days[6]

    # Every day the position was held (days 1..6 inclusive) has a nonzero
    # leg in the RAW (unsliced) daily_returns -- sanity check on the
    # engine's own contract before testing the reconciliation.
    for d in days[1:7]:
        assert daily_returns.loc[d] != 0.0

    real_trades, real_daily_returns = intraday_holdout.reconcile_holdout_window(
        trades_df, daily_returns, df, holdout_start, holdout_end,
        cost_bps=0.0, tax_haircut_rate=0.0, direction="long")

    # The TRADE-LEVEL side: this trade SPANS the boundary (entered day 1,
    # padding; exits day 6, real holdout), so its net_ret must be RE-PRICED
    # to reflect only the holdout-attributable portion of the move -- see
    # module docstring. Compute the expected re-priced value independently,
    # via the documented formula (synthetic entry = holdout_start's first
    # bar's Open), rather than hardcoding a number.
    assert len(real_trades) == 1
    holdout_start_bar = df.loc[df.index.date == days[4]].iloc[0]
    synthetic_entry_px = float(holdout_start_bar["Open"])
    expected_net_ret = trade["exit_px"] / synthetic_entry_px - 1.0  # long, cost=0, tax=0
    real_trade = real_trades.iloc[0]
    assert real_trade["net_ret"] == pytest.approx(expected_net_ret)
    assert real_trade["gross_ret"] == pytest.approx(expected_net_ret)
    # Genuinely DIFFERENT from the original (un-reconciled) trade's net_ret
    # -- the whole point of the fix: the original number silently included
    # the padding-period portion of the move.
    assert real_trade["net_ret"] != pytest.approx(trade["net_ret"])
    assert real_trade["entry_px"] == pytest.approx(synthetic_entry_px)
    assert real_trade["entry_datetime"] == df.loc[df.index.date == days[4]].index[0]
    assert real_trade["exit_px"] == pytest.approx(trade["exit_px"])  # exit is untouched
    assert real_trade["held_days"] == 3  # days 4, 5, 6

    # The DAILY_RETURNS side: real holdout is days 4..9 (6 days). The
    # position was held through days 4 (synthetic entry), 5 (interior), 6
    # (exit) -- each must carry a genuine, nonzero, DISTINCT mark. Beyond
    # the trade-level re-pricing, day 4's and day 6's marks ALSO need
    # re-pricing here (see module docstring, "daily_returns ALSO needs
    # correcting") -- only day 5 (a TRUE interior day of both the original
    # and the re-priced trade) is expected to stay numerically identical to
    # the raw engine output. Compute the expected values independently via
    # the SAME documented helpers reconcile_holdout_window itself uses, to
    # verify correct WIRING rather than re-deriving the arithmetic by hand.
    new_days, new_ref = intraday_holdout._touched_day_reference_prices(
        df, days[4], days[6], float(trade["exit_px"]))
    expected_legs = intraday_holdout._apply_net_ret_plug(
        intraday_holdout._per_day_gross_legs(
            synthetic_entry_px, new_days, new_ref, "long"),
        new_days, "long", expected_net_ret)

    assert list(real_daily_returns.index) == days[4:10]
    held_days = [days[4], days[5], days[6]]
    for d in held_days:
        sliced_value = real_daily_returns.loc[d]
        assert sliced_value == pytest.approx(expected_legs[d])
        assert sliced_value != 0.0
    # day 5 is a true interior day in BOTH the original and re-priced walk
    # (its leg is a ratio against the PREVIOUS day's own mark, independent
    # of which entry basis is used) -- it must be numerically unchanged
    # from the raw engine output, unlike days 4 and 6.
    assert real_daily_returns.loc[days[5]] == pytest.approx(daily_returns.loc[days[5]])
    # Distinct values -- not a lump sum copy-pasted across days.
    assert len({real_daily_returns.loc[d] for d in held_days}) == 3
    # Days after the position closed (7, 8, 9) are genuinely untouched (0.0)
    # -- confirms we are not fabricating activity either.
    for d in days[7:10]:
        assert real_daily_returns.loc[d] == 0.0


def test_trade_entirely_within_padding_is_pure_warmup_and_dropped():
    df = _minute_frame(n_days=10)
    days = sorted(pd.unique(df.index.date))
    holdout_start = days[4].isoformat()
    holdout_end = days[9].isoformat()

    # Opens day 0, closes day 1 -- entirely inside padding (days 0-3).
    signal_fn = _entry_exit_signal(entry_exec_i=1, exit_exec_i=4)
    trades_df, daily_returns = run_intraday_backtest(
        df, signal_fn, max_hold_days=20, direction="long")
    assert len(trades_df) == 1

    real_trades, real_daily_returns = intraday_holdout.reconcile_holdout_window(
        trades_df, daily_returns, df, holdout_start, holdout_end, cost_bps=0.0)

    assert len(real_trades) == 0
    # None of the real holdout days show any activity from this trade.
    assert (real_daily_returns == 0.0).all()


def test_exit_exactly_on_holdout_start_counts_as_real_holdout_activity():
    """Deliberate divergence from batch3's own boundary convention -- see
    module docstring. At this DAY-level granularity, a trade exiting on the
    real holdout window's FIRST day had genuine intraday activity that day,
    inside real holdout, so it must be counted, not dropped."""
    df = _minute_frame(n_days=10)
    days = sorted(pd.unique(df.index.date))
    holdout_start = days[4].isoformat()
    holdout_end = days[9].isoformat()

    # Opens day 2 (padding), exits day 4 -- exactly holdout_start.
    entry_exec_i = 6    # day 2, bar 0
    exit_exec_i = 12    # day 4, bar 0
    signal_fn = _entry_exit_signal(entry_exec_i, exit_exec_i)
    trades_df, daily_returns = run_intraday_backtest(
        df, signal_fn, max_hold_days=20, direction="long")
    assert trades_df.iloc[0]["exit_datetime"].date() == days[4]

    real_trades, real_daily_returns = intraday_holdout.reconcile_holdout_window(
        trades_df, daily_returns, df, holdout_start, holdout_end, cost_bps=0.0)

    assert len(real_trades) == 1
    assert real_daily_returns.loc[days[4]] != 0.0
    # Re-priced (this trade spans the boundary): entry and exit must NOT be
    # the same synthetic/real price pair despite exiting on holdout_start's
    # own day -- see module docstring for why this stays safe (Open-of-
    # first-bar vs. Close-of-exit-bar are genuinely different prices).
    real_trade = real_trades.iloc[0]
    assert real_trade["entry_px"] != pytest.approx(real_trade["exit_px"])
    assert real_trade["net_ret"] != 0.0


def test_exit_exactly_on_holdout_end_is_included():
    df = _minute_frame(n_days=10)
    days = sorted(pd.unique(df.index.date))
    holdout_start = days[4].isoformat()
    holdout_end = days[9].isoformat()

    entry_exec_i = 24   # day 8, bar 0
    exit_exec_i = 27    # day 9, bar 0 (== holdout_end)
    signal_fn = _entry_exit_signal(entry_exec_i, exit_exec_i)
    trades_df, daily_returns = run_intraday_backtest(
        df, signal_fn, max_hold_days=20, direction="long")
    assert trades_df.iloc[0]["exit_datetime"].date() == days[9]

    real_trades, _ = intraday_holdout.reconcile_holdout_window(
        trades_df, daily_returns, df, holdout_start, holdout_end, cost_bps=0.0)
    assert len(real_trades) == 1
    # Entry (day 8) is >= holdout_start (day 4) -- fully inside the holdout
    # window, NOT a spanning trade, so it must be left completely
    # unchanged (no re-pricing).
    assert real_trades.iloc[0]["net_ret"] == pytest.approx(
        trades_df.iloc[0]["net_ret"])
    assert real_trades.iloc[0]["entry_px"] == pytest.approx(
        trades_df.iloc[0]["entry_px"])


def test_short_direction_held_into_holdout_also_gets_genuine_marks():
    """The engine's SHORT convention uses a fixed-basis (from entry_px), not
    a rolling previous-close basis -- confirm the slice is still honest for
    a short position spanning the padding/holdout boundary."""
    df = _minute_frame(n_days=10)
    days = sorted(pd.unique(df.index.date))
    holdout_start = days[4].isoformat()
    holdout_end = days[9].isoformat()

    entry_exec_i = 3
    exit_exec_i = 18
    signal_fn = _entry_exit_signal(entry_exec_i, exit_exec_i)
    trades_df, daily_returns = run_intraday_backtest(
        df, signal_fn, max_hold_days=20, direction="short")

    real_trades, real_daily_returns = intraday_holdout.reconcile_holdout_window(
        trades_df, daily_returns, df, holdout_start, holdout_end,
        cost_bps=0.0, direction="short")

    assert len(real_trades) == 1

    # SHORT re-pricing uses the "-long_ret" branch -- verify it independently
    # against the documented formula, not just that it runs without error.
    holdout_start_bar = df.loc[df.index.date == days[4]].iloc[0]
    synthetic_entry_px = float(holdout_start_bar["Open"])
    orig_trade = trades_df.iloc[0]
    expected_long_ret = orig_trade["exit_px"] / synthetic_entry_px - 1.0
    expected_net_ret = -expected_long_ret  # short, cost=0, tax=0
    real_trade = real_trades.iloc[0]
    assert real_trade["net_ret"] == pytest.approx(expected_net_ret)
    assert real_trade["net_ret"] != pytest.approx(orig_trade["net_ret"])

    # daily_returns: SHORT uses a FIXED entry_px basis for EVERY leg
    # (unlike long's rolling previous-day basis), so unlike the long case,
    # even the "true interior" day (day 5) needs re-pricing here -- verify
    # against the SAME documented helpers reconcile_holdout_window itself
    # uses, and confirm none of the three held days matches the raw
    # (padding-basis) engine output.
    new_days, new_ref = intraday_holdout._touched_day_reference_prices(
        df, days[4], days[6], float(orig_trade["exit_px"]))
    expected_legs = intraday_holdout._apply_net_ret_plug(
        intraday_holdout._per_day_gross_legs(
            synthetic_entry_px, new_days, new_ref, "short"),
        new_days, "short", expected_net_ret)

    held_days = [days[4], days[5], days[6]]
    for d in held_days:
        assert real_daily_returns.loc[d] != 0.0
        assert real_daily_returns.loc[d] == pytest.approx(expected_legs[d])
        assert real_daily_returns.loc[d] != pytest.approx(daily_returns.loc[d])


def test_empty_trades_and_daily_returns_reconcile_cleanly():
    empty_trades = pd.DataFrame({
        "entry_datetime": pd.Series(dtype="datetime64[ns, America/New_York]"),
        "exit_datetime": pd.Series(dtype="datetime64[ns, America/New_York]"),
        "entry_px": pd.Series(dtype="float64"),
        "exit_px": pd.Series(dtype="float64"),
        "held_days": pd.Series(dtype="int64"),
        "reason": pd.Series(dtype="object"),
        "gross_ret": pd.Series(dtype="float64"),
        "net_ret": pd.Series(dtype="float64"),
    })
    empty_daily = pd.Series(dtype="float64")
    real_trades, real_daily = intraday_holdout.reconcile_holdout_window(
        empty_trades, empty_daily, pd.DataFrame(), "2024-01-08", "2024-01-15",
        cost_bps=0.0)
    assert len(real_trades) == 0
    assert len(real_daily) == 0


def test_reconcile_raises_when_start_after_end():
    empty_trades = pd.DataFrame({
        "entry_datetime": pd.Series(dtype="datetime64[ns, America/New_York]"),
        "exit_datetime": pd.Series(dtype="datetime64[ns, America/New_York]"),
        "entry_px": pd.Series(dtype="float64"), "exit_px": pd.Series(dtype="float64"),
        "held_days": pd.Series(dtype="int64"), "reason": pd.Series(dtype="object"),
        "gross_ret": pd.Series(dtype="float64"), "net_ret": pd.Series(dtype="float64"),
    })
    with pytest.raises(ValueError, match="after"):
        intraday_holdout.reconcile_holdout_window(
            empty_trades, pd.Series(dtype="float64"), pd.DataFrame(),
            "2024-01-15", "2024-01-08", cost_bps=0.0)


def test_reconcile_raises_when_spanning_trade_but_bars_dont_cover_holdout():
    df = _minute_frame(n_days=10)
    days = sorted(pd.unique(df.index.date))
    holdout_start = days[4].isoformat()
    holdout_end = days[9].isoformat()
    signal_fn = _entry_exit_signal(entry_exec_i=3, exit_exec_i=18)
    trades_df, daily_returns = run_intraday_backtest(
        df, signal_fn, max_hold_days=20, direction="long")

    # `bars` deliberately truncated BEFORE the real holdout window even
    # starts -- simulates a caller bug (fetched too little data) rather
    # than the intraday_data.py fetch contract, which always fetches
    # through holdout_end.
    truncated_bars = df.loc[df.index.date < days[4]]
    with pytest.raises(ValueError, match="no rows"):
        intraday_holdout.reconcile_holdout_window(
            trades_df, daily_returns, truncated_bars, holdout_start,
            holdout_end, cost_bps=0.0)


def test_reconcile_raises_when_mismatched_bars_would_put_synthetic_entry_after_exit():
    """Regression test: a caller passing
    a `bars` frame that does NOT actually match `trades_df` -- specifically
    one missing the EARLY days of the real holdout window where a spanning
    trade actually exits (e.g. accidentally handing in a frame fetched from
    the wrong start date) -- must fail loudly, not silently produce a
    temporally-impossible (synthetic entry chronologically AFTER the real
    exit) result."""
    df = _minute_frame(n_days=10)
    days = sorted(pd.unique(df.index.date))
    holdout_start = days[4].isoformat()
    holdout_end = days[9].isoformat()

    # Spans the boundary: opens day 1 (padding), exits day 4 -- the very
    # first real-holdout day.
    signal_fn = _entry_exit_signal(entry_exec_i=3, exit_exec_i=12)
    trades_df, daily_returns = run_intraday_backtest(
        df, signal_fn, max_hold_days=20, direction="long")
    assert trades_df.iloc[0]["exit_datetime"].date() == days[4]

    # `bars` mismatched: starts at day 6 instead of covering
    # [holdout_start, holdout_end] fully -- still produces a NON-empty
    # `holdout_bars` slice (days 6-9 all satisfy date>=holdout_start), so
    # the earlier "no rows at all" guard does NOT fire; its first row
    # (day 6) is chronologically AFTER the trade's real exit (day 4),
    # which is exactly the mismatch this guard exists to catch.
    mismatched_bars = df.loc[df.index.date >= days[6]]
    with pytest.raises(ValueError, match="before the synthetic holdout entry"):
        intraday_holdout.reconcile_holdout_window(
            trades_df, daily_returns, mismatched_bars, holdout_start,
            holdout_end, cost_bps=0.0)


@pytest.mark.parametrize("direction", ["long", "short"])
def test_reconcile_never_produces_synthetic_entry_after_real_exit(direction):
    """Permanent regression guard (sweeps every entry/exit bar pair across
    multiple holdout_start offsets for both directions): when `bars` genuinely matches `trades_df` (the
    real, honest calling contract), a spanning trade's synthetic entry must
    never land chronologically after its own real exit."""
    n_days = 12
    df = _minute_frame(n_days=n_days)
    days = sorted(pd.unique(df.index.date))
    n_bars = len(df)
    checked = 0

    for holdout_start_day_idx in range(2, n_days - 1):
        holdout_start = days[holdout_start_day_idx].isoformat()
        holdout_end = days[-1].isoformat()
        for entry_exec_i in range(1, holdout_start_day_idx * 3, 2):
            for exit_exec_i in range(entry_exec_i, n_bars, 3):
                signal_fn = _entry_exit_signal(entry_exec_i, exit_exec_i)
                trades_df, daily_returns = run_intraday_backtest(
                    df, signal_fn, cost_bps=1.0, tax_haircut_rate=0.1,
                    max_hold_days=200, direction=direction)
                if len(trades_df) != 1:
                    continue
                exit_date = trades_df.iloc[0]["exit_datetime"].date()
                if exit_date < days[holdout_start_day_idx]:
                    continue
                real_trades, _ = intraday_holdout.reconcile_holdout_window(
                    trades_df, daily_returns, df, holdout_start, holdout_end,
                    cost_bps=1.0, tax_haircut_rate=0.1, direction=direction)
                if len(real_trades) == 0:
                    continue
                checked += 1
                assert (real_trades.iloc[0]["entry_datetime"]
                        <= trades_df.iloc[0]["exit_datetime"])
                assert real_trades.iloc[0]["held_days"] >= 1
    assert checked > 20, "test assumption broken: too few spanning scenarios exercised"


# ---------------------------------------------------------------------------
# pool_intraday_trades
# ---------------------------------------------------------------------------

def test_pool_intraday_trades_tags_and_sorts_chronologically():
    df_a = _minute_frame(n_days=5, start_date="2024-01-02")
    df_b = _minute_frame(n_days=5, start_date="2024-01-02")
    signal_a = _entry_exit_signal(entry_exec_i=6, exit_exec_i=9)   # later
    signal_b = _entry_exit_signal(entry_exec_i=3, exit_exec_i=6)   # earlier
    trades_a, _ = run_intraday_backtest(df_a, signal_a, max_hold_days=20)
    trades_b, _ = run_intraday_backtest(df_b, signal_b, max_hold_days=20)

    pooled = intraday_holdout.pool_intraday_trades({"AAA": trades_a, "BBB": trades_b})
    assert list(pooled["ticker"]) == ["BBB", "AAA"]  # BBB entered first
    assert len(pooled) == 2
    assert set(pooled.columns) >= {"entry_datetime", "exit_datetime", "net_ret", "ticker"}


def test_pool_intraday_trades_skips_zero_trade_tickers():
    df = _minute_frame(n_days=5)
    no_trade_signal = lambda frame, i: SignalResult(entry=False, exit=False)
    trades_empty, _ = run_intraday_backtest(df, no_trade_signal, max_hold_days=20)
    signal_b = _entry_exit_signal(entry_exec_i=3, exit_exec_i=6)
    trades_b, _ = run_intraday_backtest(df, signal_b, max_hold_days=20)

    pooled = intraday_holdout.pool_intraday_trades(
        {"EMPTY": trades_empty, "BBB": trades_b})
    assert list(pooled["ticker"]) == ["BBB"]


def test_pool_intraday_trades_all_empty_returns_empty_frame_with_ticker_column():
    df = _minute_frame(n_days=5)
    no_trade_signal = lambda frame, i: SignalResult(entry=False, exit=False)
    trades_empty, _ = run_intraday_backtest(df, no_trade_signal, max_hold_days=20)

    pooled = intraday_holdout.pool_intraday_trades({"AAA": trades_empty})
    assert len(pooled) == 0
    assert "ticker" in pooled.columns
    assert "net_ret" in pooled.columns


def test_pool_intraday_trades_raises_on_empty_dict():
    with pytest.raises(ValueError, match="empty"):
        intraday_holdout.pool_intraday_trades({})


# ---------------------------------------------------------------------------
# pool_swing_daily_returns
# ---------------------------------------------------------------------------

def test_pool_swing_daily_returns_sums_not_averages_on_union():
    from datetime import date
    s_a = pd.Series([0.01, 0.02], index=pd.Index([date(2024, 1, 2), date(2024, 1, 3)]))
    s_b = pd.Series([0.03], index=pd.Index([date(2024, 1, 3)]))
    combined = intraday_holdout.pool_swing_daily_returns({"AAA": s_a, "BBB": s_b})
    assert combined.loc[date(2024, 1, 2)] == pytest.approx(0.01)
    assert combined.loc[date(2024, 1, 3)] == pytest.approx(0.05)  # summed, not averaged
    assert len(combined) == 2


def test_pool_swing_daily_returns_raises_on_empty_dict():
    with pytest.raises(ValueError, match="empty"):
        intraday_holdout.pool_swing_daily_returns({})


# ---------------------------------------------------------------------------
# intraday_trade_stats
# ---------------------------------------------------------------------------

def test_intraday_trade_stats_zero_trades_returns_fallback():
    empty = pd.DataFrame({"net_ret": pd.Series(dtype="float64")})
    stats = intraday_holdout.intraday_trade_stats(empty)
    assert stats == {"trade_count": 0, "net_edge_bps": 0.0, "sharpe": 0.0,
                      "skew": 0.0, "kurtosis": 3.0}


def test_intraday_trade_stats_matches_hand_computed_values():
    returns = [0.01, -0.005, 0.02, 0.0, 0.015, -0.01, 0.03, 0.005, -0.002, 0.012,
               0.008, -0.003, 0.017, 0.001, 0.009]  # 15 trades, MIN_TRADES_FOR_SCORING
    df = pd.DataFrame({"net_ret": returns})
    stats = intraday_holdout.intraday_trade_stats(df)
    arr = np.array(returns)
    assert stats["trade_count"] == 15
    assert stats["net_edge_bps"] == pytest.approx(float(np.mean(arr) * 1e4))
    std = float(np.std(arr, ddof=1))
    assert stats["sharpe"] == pytest.approx(float(np.mean(arr) / std))
    assert stats["kurtosis"] >= 1.0 + stats["skew"] ** 2 - 1e-9


def test_intraday_trade_stats_single_trade_fallback():
    df = pd.DataFrame({"net_ret": [0.02]})
    stats = intraday_holdout.intraday_trade_stats(df)
    assert stats["trade_count"] == 1
    assert stats["sharpe"] == 0.0
    assert stats["skew"] == 0.0
    assert stats["kurtosis"] == 3.0
    assert stats["net_edge_bps"] == pytest.approx(200.0)


def test_intraday_trade_stats_zero_variance_falls_back_safely():
    df = pd.DataFrame({"net_ret": [0.01, 0.01, 0.01, 0.01]})
    stats = intraday_holdout.intraday_trade_stats(df)
    assert stats["sharpe"] == 0.0
    assert stats["skew"] == 0.0
    assert stats["kurtosis"] == 3.0


def test_intraday_trade_stats_kurtosis_floor_never_violates_dsr_precondition():
    """The whole reason this helper mirrors _trade_stats instead of
    pooled_trade_stats: the output must never violate
    dsr.probabilistic_sharpe_ratio's kurtosis >= 1 + skew**2 precondition."""
    from mining import dsr
    rng = np.random.default_rng(7)
    for _ in range(25):
        n = rng.integers(2, 40)
        returns = rng.normal(0, 0.01, size=n)
        # Force some asymmetry / fat tails to stress the floor.
        returns[0] += rng.choice([-1, 1]) * rng.uniform(0.05, 0.5)
        df = pd.DataFrame({"net_ret": returns})
        stats = intraday_holdout.intraday_trade_stats(df)
        if stats["trade_count"] >= 2:
            # Must not raise.
            dsr.probabilistic_sharpe_ratio(
                sr=stats["sharpe"], sr_benchmark=0.0, n=stats["trade_count"],
                skew=stats["skew"], kurtosis=stats["kurtosis"])


def test_intraday_trade_stats_raises_on_non_finite_returns():
    df = pd.DataFrame({"net_ret": [0.01, float("nan"), 0.02]})
    with pytest.raises(ValueError, match="finite"):
        intraday_holdout.intraday_trade_stats(df)
