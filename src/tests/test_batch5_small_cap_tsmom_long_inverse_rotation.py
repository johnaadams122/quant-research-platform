"""Tests for src/mining/batch5/small_cap_tsmom_long_inverse_rotation.py.

Synthetic frames only (no network). Monthly closes are hand-built so the
1-/3-/12-month sign votes at each month-end are known in advance; the
daily path inside a month is a straight-line interpolation that lands the
month's LAST session exactly on the designed month-end close.
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

from mining.batch5 import small_cap_tsmom_long_inverse_rotation as mod
from mining.rotation_engine import run_two_asset_rotation

MODULE_PATH = os.path.join(_ROOT, "src", "mining", "batch5",
                           "small_cap_tsmom_long_inverse_rotation.py")


# ------------------------------------------------------------ synthetic data

def _calendar(start: str, n_months: int) -> pd.DatetimeIndex:
    """Business days covering ``n_months`` whole calendar months from the
    first business day of ``start``'s month."""
    first = pd.Timestamp(start).to_period("M").to_timestamp(how="start")
    last = (first + pd.DateOffset(months=n_months) - pd.Timedelta(days=1))
    return pd.bdate_range(first, last)


def _frame_from_month_ends(index: pd.DatetimeIndex, month_end_closes) -> pd.DataFrame:
    """OHLC frame whose LAST session of calendar month j closes exactly at
    ``month_end_closes[j]``; within a month the close moves linearly from
    the prior month-end close. Open_t == Close_{t-1}; High/Low bracket."""
    periods = index.to_period("M")
    unique = list(periods.unique())
    if len(unique) != len(month_end_closes):
        raise ValueError("one designed close per calendar month is required")
    close = np.empty(len(index), dtype=float)
    prev = float(month_end_closes[0]) * 0.98
    for period, end_value in zip(unique, month_end_closes):
        positions = np.flatnonzero(periods == period)
        steps = np.linspace(prev, float(end_value), len(positions) + 1)[1:]
        close[positions] = steps
        prev = float(end_value)
    open_ = np.concatenate([[close[0] * 0.999], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close}, index=index)


def _inverse_like(frame: pd.DataFrame) -> pd.DataFrame:
    """A positive 'inverse fund' path on the same calendar (its values never
    enter the decision; only its calendar matters)."""
    rets = frame["Close"].pct_change().fillna(0.0).to_numpy()
    close = 50.0 * np.cumprod(1.0 - rets)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.001,
                         "Low": np.minimum(open_, close) * 0.999,
                         "Close": close}, index=frame.index)


# Designed IWM month-end closes (months 0..19). Sign votes (1m / 3m / 12m) at
# each decidable month-end, hand-derived:
#   m12=112 vs 111,109,100 -> +,+,+ -> IWM
#   m13=111 vs 112,110,101 -> -,+,+ -> IWM
#   m14=105 vs 111,111,102 -> -,-,+ -> RWM
#   m15=106 vs 105,112,103 -> +,-,+ -> IWM
#   m16=103 vs 106,111,104 -> -,-,- -> RWM
#   m17=104 vs 103,105,105 -> +,-,- -> RWM
#   m18=104 vs 104,106,106 -> 0(-),-,- -> RWM
#   m19=104 vs 104,103,107 -> 0(-),+,- -> RWM   (zero must vote -1: else IWM)
#   m20 is the calendar's final month: its last bar is never a decision date.
MONTH_ENDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
              112, 111, 105, 106, 103, 104, 104, 104, 105]
EXPECTED_BY_MONTH = {12: "IWM", 13: "IWM", 14: "RWM", 15: "IWM", 16: "RWM",
                     17: "RWM", 18: "RWM", 19: "RWM"}


def _designed_frames():
    index = _calendar("2015-01-01", len(MONTH_ENDS))
    iwm = _frame_from_month_ends(index, MONTH_ENDS)
    return {"IWM": iwm, "RWM": _inverse_like(iwm)}


def _month_end_sessions(index: pd.DatetimeIndex) -> list:
    """Independent month-end derivation: last session per calendar month,
    excluding the final session of the calendar."""
    periods = index.to_period("M")
    out = []
    for period in periods.unique()[:-1]:
        out.append(index[np.flatnonzero(periods == period)[-1]])
    return out


def _run_targets(target_fn, engine_frames, upto=None):
    """Mimic the engine: call target_fn with frames truncated through each
    bar of the shared calendar; return {date: result}."""
    calendar = next(iter(engine_frames.values())).index
    n = len(calendar) if upto is None else upto
    results = {}
    for t in range(n):
        truncated = {key: frame.iloc[:t + 1].copy(deep=True)
                     for key, frame in engine_frames.items()}
        results[calendar[t]] = target_fn(truncated)
    return results


# ---------------------------------------------------------------- contract

# ------------------------------------------------------------------- rule

def test_majority_sign_rule_fires_where_designed_and_holds_between():
    frames = _designed_frames()
    target_fn = mod.make_target_fn(frames)
    results = _run_targets(target_fn, frames)
    index = frames["IWM"].index
    month_ends = _month_end_sessions(index)
    assert list(target_fn.decision_dates) == month_ends
    periods = index.to_period("M")
    unique = list(periods.unique())
    for month_index, expected in EXPECTED_BY_MONTH.items():
        me = month_ends[month_index]
        assert results[me] == expected, (month_index, me)
        # The bar BEFORE the month-end still carries the previous month.
        prev_bar = index[index.get_loc(me) - 1]
        prior = EXPECTED_BY_MONTH.get(month_index - 1)
        assert results[prev_bar] == prior, (month_index, prev_bar)
        # Every bar of the FOLLOWING month holds this month's decision
        # (up to but excluding that month's own month-end).
        next_period = unique[month_index + 1] if month_index + 1 < len(unique) else None
        if next_period is not None:
            following = index[periods == next_period]
            for bar in following[:-1]:
                assert results[bar] == expected, (month_index, bar)
    # No decision before 12 prior month-end closes exist.
    for bar in index[index < month_ends[12]]:
        assert results[bar] is None, bar


def test_zero_return_votes_negative():
    # Month 19: 1m return exactly 0, 3m positive, 12m negative -> RWM.
    # If zero voted +1 the majority would flip to IWM.
    assert mod.composite_target(104.0, [104.0, 103.0, 107.0], "IWM", "RWM") == "RWM"
    assert mod.composite_target(104.0, [103.0, 100.0, 107.0], "IWM", "RWM") == "IWM"
    # 1m zero, 3m zero, 12m positive: two zero votes -> RWM.
    assert mod.composite_target(104.0, [104.0, 104.0, 100.0], "IWM", "RWM") == "RWM"
    assert mod.composite_target(104.0, [None, 103.0, 100.0], "IWM", "RWM") is None


def test_iwm_history_before_rwm_inception_feeds_the_lookbacks():
    """Mirrors the real in-sample start: RWM begins years after IWM, and the
    FIRST shared month-end must already be decidable from IWM's own earlier
    month-end closes (the candidate's 'in-sample starts at RWM's first
    month-end')."""
    index = _calendar("2015-01-01", len(MONTH_ENDS))
    iwm = _frame_from_month_ends(index, MONTH_ENDS)
    rwm_full = _inverse_like(iwm)
    rwm = rwm_full.loc[rwm_full.index >= pd.Timestamp("2016-03-10")]   # month 14, mid-month
    frames = {"IWM": iwm, "RWM": rwm}
    target_fn = mod.make_target_fn(frames)
    shared = iwm.index.intersection(rwm.index)
    assert target_fn.decision_dates[0] == _month_end_sessions(shared)[0]
    assert target_fn.decision_dates[0].to_period("M") == pd.Period("2016-03", "M")
    engine_frames = {"IWM": iwm.loc[shared], "RWM": rwm.loc[shared]}
    results = _run_targets(target_fn, engine_frames)
    first_me = target_fn.decision_dates[0]
    assert results[first_me] == EXPECTED_BY_MONTH[14] == "RWM"
    # Before the first shared month-end: no standing decision -> flat.
    for bar in shared[shared < first_me]:
        assert results[bar] is None
    # Later months follow the designed table exactly.
    for month_index in range(15, 20):
        me = _month_end_sessions(index)[month_index]
        assert results[me] == EXPECTED_BY_MONTH[month_index]


def test_undecidable_month_end_keeps_the_standing_target():
    dates = _calendar("2015-01-01", 17)
    close = pd.Series(np.linspace(100.0, 130.0, len(dates)), index=dates)
    # Remove every bar of 2015-03 from IWM's own history: the 2016-03
    # month-end needs M[m-12] = 2015-03, which no longer exists.
    close = close[close.index.to_period("M") != pd.Period("2015-03", "M")]
    assert pd.Period("2015-03", "M") not in mod.month_end_closes(close).index
    decisions = mod.decision_series(close, close.index)
    by_period = {date.to_period("M"): value for date, value in decisions.items()}
    for month in ("2015-01", "2015-02", "2015-04", "2015-12"):
        assert by_period[pd.Period(month, "M")] is None       # no 12m anchor yet
    assert by_period[pd.Period("2016-01", "M")] == "IWM"
    assert by_period[pd.Period("2016-02", "M")] == "IWM"
    # Undecidable month-end -> the standing target carries forward.
    assert by_period[pd.Period("2016-03", "M")] == "IWM"
    assert by_period[pd.Period("2016-04", "M")] == "IWM"
    # The final month (2017-05) is never a decision date.
    assert pd.Period("2017-05", "M") not in by_period


def test_final_session_is_never_a_decision_date():
    index = _calendar("2015-01-01", 3)
    sessions = mod.shared_month_end_sessions(index)
    assert len(sessions) == 2
    assert index[-1] not in sessions
    # Truncating mid-month adds no decision date at the new final bar.
    cut = index[:len(index) - 5]
    assert list(mod.shared_month_end_sessions(cut)) == list(sessions)


# ------------------------------------------------------------- look-ahead

def test_look_ahead_probe_perturbing_future_bars_never_changes_the_past():
    frames = _designed_frames()
    index = frames["IWM"].index
    k = index.get_loc(_month_end_sessions(index)[15])      # a decision bar
    baseline = _run_targets(mod.make_target_fn(frames), frames, upto=k + 1)

    perturbed = {}
    for key, frame in frames.items():
        alt = frame.copy()
        later = alt.index > index[k]
        alt.loc[later, "Close"] = alt.loc[later, "Close"] * 1.3
        alt.loc[later, "Open"] = alt.loc[later, "Open"] * 1.3
        alt.loc[later, "High"] = alt.loc[later, "High"] * 1.45
        alt.loc[later, "Low"] = alt.loc[later, "Low"] * 1.2
        perturbed[key] = alt
    rebuilt = _run_targets(mod.make_target_fn(perturbed), perturbed, upto=k + 1)
    assert rebuilt == baseline
    assert baseline[index[k]] == "IWM"                    # month 15's decision

    # Holdout-blind variant: DROP every bar after k entirely and rebuild.
    truncated = {key: frame.iloc[:k + 1].copy() for key, frame in frames.items()}
    cut = _run_targets(mod.make_target_fn(truncated), truncated)
    # Bar k is the calendar's final session in the truncated build, so it
    # cannot be a decision date there; every EARLIER bar must agree.
    assert {d: v for d, v in cut.items() if d < index[k]} == \
           {d: v for d, v in baseline.items() if d < index[k]}


def test_factory_and_engine_must_run_on_the_same_data():
    frames = _designed_frames()
    target_fn = mod.make_target_fn(frames)
    other = {key: frame.copy() for key, frame in frames.items()}
    other["IWM"]["Close"] = other["IWM"]["Close"] * 1.01
    with pytest.raises(ValueError, match="does not match the factory"):
        target_fn({key: frame.iloc[:300] for key, frame in other.items()})
    with pytest.raises(ValueError):
        mod.make_target_fn({"IWM": frames["IWM"]})          # RWM missing


# ------------------------------------------------------------- end to end

def test_end_to_end_through_the_rotation_engine():
    # ~28 months of RWM (about 600 bars) preceded by 14 extra months of IWM
    # history so the first shared month-end is already decidable.
    n_months = 42
    index = _calendar("2010-01-01", n_months)
    # IWM: rises 1%/month for 30 months, then falls 3%/month.
    closes = [100.0 * 1.01 ** j for j in range(30)]
    closes += [closes[-1] * 0.97 ** (j + 1) for j in range(n_months - 30)]
    iwm = _frame_from_month_ends(index, closes)
    rwm_full = _inverse_like(iwm)
    rwm = rwm_full.loc[rwm_full.index >= pd.Timestamp("2011-03-01")]
    shared = iwm.index.intersection(rwm.index)
    assert 550 <= len(shared) <= 700
    frames = {"IWM": iwm, "RWM": rwm}
    target_fn = mod.make_target_fn(frames)
    engine_frames = {"IWM": iwm.loc[shared], "RWM": rwm.loc[shared]}
    trades, active = run_two_asset_rotation(
        engine_frames, target_fn,
        cost_bps_by_asset={"IWM": 2.0, "RWM": 8.0}, tax_haircut_rate=0.0)
    assert len(trades) >= 2
    assert set(trades["asset"]) == {"IWM", "RWM"}
    assert list(trades["asset"]) == ["IWM", "RWM"]        # one switch, up then down
    # The first segment opens at the session AFTER the first shared
    # month-end (decision at month-end close, fill at next open).
    first_me = target_fn.decision_dates[0]
    assert trades["entry_date"].iloc[0] == shared[shared.get_loc(first_me) + 1]
    # The switch to RWM also lands on the session after a month-end.
    switch_entry = trades["entry_date"].iloc[1]
    prior_session = shared[shared.get_loc(switch_entry) - 1]
    assert prior_session in set(target_fn.decision_dates)
    assert trades["exit_date"].iloc[0] == switch_entry           # same-bar switch
    assert trades["reason"].tolist() == ["switch", "end_of_data"]
    assert active.dropna().isin(["IWM", "RWM"]).all()
