import os
import sys

import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import cost_tax_inputs
from mining import constants


def test_compute_spread_bps_simple_round_numbers():
    assert cost_tax_inputs.compute_spread_bps(99.0, 101.0) == pytest.approx(200.0)


def test_compute_spread_bps_rejects_non_positive_mid():
    with pytest.raises(ValueError):
        cost_tax_inputs.compute_spread_bps(-1.0, -2.0)


def test_session_round_trip_averages_entry_and_exit_half_spreads():
    assert cost_tax_inputs.session_round_trip(2.0, 4.0) == pytest.approx(3.0)


def test_cost_bps_from_quotes_composes_both_formulas():
    result = cost_tax_inputs.cost_bps_from_quotes(99.0, 101.0, 199.0, 201.0)
    # entry mid=100, spread=2 -> 200bps; exit mid=200, spread=2 -> 100bps
    assert result == pytest.approx((200.0 + 100.0) / 2.0)


def test_tax_haircut_rate_zero_for_tax_advantaged_accounts():
    assert cost_tax_inputs.tax_haircut_rate_for("roth") == 0.0
    assert cost_tax_inputs.tax_haircut_rate_for("ira") == 0.0


def test_tax_haircut_rate_uses_pre_registered_constant_for_taxable():
    assert cost_tax_inputs.tax_haircut_rate_for("taxable") == constants.STCG_TAX_RATE


def test_tax_haircut_rate_rejects_unknown_account_type():
    with pytest.raises(ValueError):
        cost_tax_inputs.tax_haircut_rate_for("some_bogus_type")
