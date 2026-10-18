import math
import os
import sys

import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import constants
from mining import intraday_cost_gate


def test_passes_when_within_ceiling_and_edge_margin():
    # entry mid=100, spread=1bps -> half=0.5; exit mid=100, spread=1bps -> half=0.5
    # spread_bps = 1.0, well under the 20 bps ceiling.
    # net_edge_bps well above 3x that (3.0 bps needed, using 10.0).
    result = intraday_cost_gate.evaluate_cost_realism_gate(
        entry_bid=99.995, entry_ask=100.005,
        exit_bid=99.995, exit_ask=100.005,
        net_edge_bps=10.0,
    )
    assert result["proceed"] is True
    assert result["spread_bps"] == pytest.approx(1.0)
    assert "ceiling" in result["reason"] or "margin" in result["reason"]


def test_fails_when_spread_at_or_above_ceiling():
    # Construct a spread of EXACTLY 20.0 bps round-trip at the float
    # level (both legs identical, so round-trip == each leg's own
    # spread): bid=9990, ask=10010 -> mid=10000, ask-bid=20 ->
    # spread=(20/10000)*1e4=20.0 exactly (verified: no floating-point
    # rounding noise at these round numbers, unlike e.g. 99.9/100.1).
    bid, ask = 9990.0, 10010.0
    result = intraday_cost_gate.evaluate_cost_realism_gate(
        entry_bid=bid, entry_ask=ask,
        exit_bid=bid, exit_ask=ask,
        net_edge_bps=1000.0,  # would easily pass the edge-margin check
    )
    assert result["spread_bps"] == 20.0
    assert result["spread_bps"] == constants.INTRADAY_SPREAD_CEILING_BPS
    assert result["proceed"] is False
    assert "20" in result["reason"] or f"{constants.INTRADAY_SPREAD_CEILING_BPS}" in result["reason"]


def test_fails_when_spread_clearly_above_ceiling_reason_carries_actual_value():
    # A wide, clearly-failing spread -- reason string must carry the actual
    # measured value, not a generic code.
    result = intraday_cost_gate.evaluate_cost_realism_gate(
        entry_bid=95.0, entry_ask=105.0,  # mid=100, spread=1000bps
        exit_bid=95.0, exit_ask=105.0,
        net_edge_bps=50000.0,
    )
    assert result["proceed"] is False
    assert result["spread_bps"] == pytest.approx(1000.0)
    assert "1000.00" in result["reason"] or "1000.0" in result["reason"]


def test_passes_exactly_at_edge_over_cost_multiple_boundary():
    # spread_bps = 1.0 (well under ceiling). net_edge_bps set to exactly
    # 3.0x that -> boundary of the edge-margin check, which PASSES at
    # exactly the multiple (matches this pipeline's established >=-passes
    # convention, e.g. DSR_PASS_THRESHOLD: dsr_value >= threshold passes).
    result = intraday_cost_gate.evaluate_cost_realism_gate(
        entry_bid=99.995, entry_ask=100.005,
        exit_bid=99.995, exit_ask=100.005,
        net_edge_bps=1.0 * constants.INTRADAY_EDGE_OVER_COST_MULTIPLE,
    )
    assert result["spread_bps"] == pytest.approx(1.0)
    assert result["edge_over_cost_ratio"] == pytest.approx(
        constants.INTRADAY_EDGE_OVER_COST_MULTIPLE)
    assert result["proceed"] is True


def test_fails_when_edge_over_cost_multiple_falls_just_short():
    # spread_bps = 1.0; net_edge_bps just under 3x -> fails on margin,
    # despite comfortably passing the spread ceiling.
    result = intraday_cost_gate.evaluate_cost_realism_gate(
        entry_bid=99.995, entry_ask=100.005,
        exit_bid=99.995, exit_ask=100.005,
        net_edge_bps=2.9,
    )
    assert result["spread_bps"] == pytest.approx(1.0)
    assert result["proceed"] is False
    assert "2.9" in result["reason"]


def test_fails_edge_margin_reason_carries_actual_ratio():
    result = intraday_cost_gate.evaluate_cost_realism_gate(
        entry_bid=99.995, entry_ask=100.005,
        exit_bid=99.995, exit_ask=100.005,
        net_edge_bps=1.0,  # ratio = 1.0, well short of the 3.0 multiple
    )
    assert result["proceed"] is False
    assert result["edge_over_cost_ratio"] == pytest.approx(1.0)
    assert "1.00" in result["reason"] or "1.0" in result["reason"]


def test_rejects_nan_net_edge_bps():
    with pytest.raises(ValueError):
        intraday_cost_gate.evaluate_cost_realism_gate(
            entry_bid=99.995, entry_ask=100.005,
            exit_bid=99.995, exit_ask=100.005,
            net_edge_bps=float("nan"),
        )


def test_rejects_infinite_net_edge_bps():
    with pytest.raises(ValueError):
        intraday_cost_gate.evaluate_cost_realism_gate(
            entry_bid=99.995, entry_ask=100.005,
            exit_bid=99.995, exit_ask=100.005,
            net_edge_bps=float("inf"),
        )


def test_rejects_negative_bid_and_ask_before_reaching_compute_spread_bps():
    # The gate's own upfront positivity check catches this before it
    # would ever reach cost_tax_inputs.compute_spread_bps's non-positive
    # mid check -- still a ValueError either way, never swallowed.
    with pytest.raises(ValueError):
        intraday_cost_gate.evaluate_cost_realism_gate(
            entry_bid=-1.0, entry_ask=-2.0,
            exit_bid=99.995, exit_ask=100.005,
            net_edge_bps=10.0,
        )


# --- Quote validation: a crossed, NaN/inf, or non-positive bid/ask is bad
# data, not a real market state, and this pipeline fails closed on bad
# data rather than let it silently produce a number. A crossed quote in
# particular matters: its negative spread_bps would otherwise sail
# through both threshold checks (always < the ceiling, and comparing
# net_edge_bps against a negative "cost") and incorrectly ADMIT a
# candidate.

def test_rejects_crossed_entry_quote():
    with pytest.raises(ValueError, match="entry"):
        intraday_cost_gate.evaluate_cost_realism_gate(
            entry_bid=100.10, entry_ask=99.90,  # crossed: bid > ask
            exit_bid=99.995, exit_ask=100.005,
            net_edge_bps=10.0,
        )


def test_rejects_crossed_exit_quote():
    with pytest.raises(ValueError, match="exit"):
        intraday_cost_gate.evaluate_cost_realism_gate(
            entry_bid=99.995, entry_ask=100.005,
            exit_bid=100.10, exit_ask=99.90,  # crossed: bid > ask
            net_edge_bps=10.0,
        )


@pytest.mark.parametrize("field", [
    "entry_bid", "entry_ask", "exit_bid", "exit_ask",
])
def test_rejects_nan_quote_field(field):
    kwargs = dict(entry_bid=99.995, entry_ask=100.005,
                  exit_bid=99.995, exit_ask=100.005, net_edge_bps=10.0)
    kwargs[field] = float("nan")
    with pytest.raises(ValueError):
        intraday_cost_gate.evaluate_cost_realism_gate(**kwargs)


@pytest.mark.parametrize("field", [
    "entry_bid", "entry_ask", "exit_bid", "exit_ask",
])
def test_rejects_infinite_quote_field(field):
    kwargs = dict(entry_bid=99.995, entry_ask=100.005,
                  exit_bid=99.995, exit_ask=100.005, net_edge_bps=10.0)
    kwargs[field] = float("inf")
    with pytest.raises(ValueError):
        intraday_cost_gate.evaluate_cost_realism_gate(**kwargs)


@pytest.mark.parametrize("field", [
    "entry_bid", "entry_ask", "exit_bid", "exit_ask",
])
def test_rejects_zero_quote_field(field):
    kwargs = dict(entry_bid=99.995, entry_ask=100.005,
                  exit_bid=99.995, exit_ask=100.005, net_edge_bps=10.0)
    kwargs[field] = 0.0
    with pytest.raises(ValueError):
        intraday_cost_gate.evaluate_cost_realism_gate(**kwargs)


@pytest.mark.parametrize("field", [
    "entry_bid", "entry_ask", "exit_bid", "exit_ask",
])
def test_rejects_negative_quote_field(field):
    kwargs = dict(entry_bid=99.995, entry_ask=100.005,
                  exit_bid=99.995, exit_ask=100.005, net_edge_bps=10.0)
    kwargs[field] = -5.0
    with pytest.raises(ValueError):
        intraday_cost_gate.evaluate_cost_realism_gate(**kwargs)


def test_crossed_quote_error_message_names_the_side_and_actual_values():
    with pytest.raises(ValueError) as excinfo:
        intraday_cost_gate.evaluate_cost_realism_gate(
            entry_bid=100.10, entry_ask=99.90,
            exit_bid=99.995, exit_ask=100.005,
            net_edge_bps=10.0,
        )
    message = str(excinfo.value)
    assert "entry" in message
    assert "100.1" in message
    assert "99.9" in message
