import json
import os
import sys

import pytest

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mining import catalog as catalog_mod

_SEED_PATH = os.path.join(_SRC, "..", "data", "mining", "catalog.json")


def _valid_entry(**overrides):
    base = {
        "family_id": "x", "name": "x", "mechanism": "x", "tickers": ["QQQ"],
        "parameters": {}, "account_types": ["roth"], "status": "eligible",
        "source": "x", "added_date": "2026-08-27",
    }
    base.update(overrides)
    return base


def test_load_catalog_reads_real_seed_file():
    catalog = catalog_mod.load_catalog(_SEED_PATH)
    # The total is pinned here so that additions and removals are deliberate.
    assert len(catalog) == 183
    assert all("family_id" in e for e in catalog)


def test_seed_catalog_original_12_entries_stay_ruled_out():
    """The 12 seed entries are pinned as ruled_out, so none can be silently
    flipped eligible."""
    catalog = catalog_mod.load_catalog(_SEED_PATH)
    original_ids = {
        "long_mean_reversion_ibs_rsi2", "dual_momentum_taa_trend",
        "gayed_200dma_letf_rotation", "turn_of_month", "pre_fomc_drift",
        "qqq_tqqq_orb", "nine_sig_tqqq_value_averaging", "overnight_drift",
        "tiny_titans_microcap", "options_wheel_csp_cc_single_name",
        "long_option_shapes_0dte_tail", "stocks_in_play_orb",
    }
    by_id = {e["family_id"]: e for e in catalog}
    assert original_ids <= set(by_id)
    assert all(by_id[fid]["status"] == "ruled_out" for fid in original_ids)


def test_seed_catalog_pins_the_full_set_of_eligible_entries():
    """The catalog holds exactly this set of eligible families (seasonal and
    calendar rules, synthetic-leverage rotations, cross-asset regime filters,
    intraday continuation setups, and benchmark-plus-overlay accounts); the list
    below pins the full set so that any addition or status flip is deliberate."""
    catalog = catalog_mod.load_catalog(_SEED_PATH)
    eligible_ids = {e["family_id"] for e in catalog_mod.eligible_entries(catalog)}
    assert eligible_ids == {
        "aaii_cash_allocation_contrarian",
        "accrual_anomaly_sloan",
        "analyst_rating_upgrade_momentum",
        "anchored_vwap_reclaim_continuation",
        "avwap_band_extension_reversion",
        "avwap_swing_pivot_reclaim_breakout",
        "betting_against_beta_low_vol",
        "bitcoin_extreme_decline_spillover_riskoff",
        "blue_chip_reversal_no_capex_control",
        "buyable_gap_up_continuation",
        "buyback_yield_repurchase_anomaly",
        "capex_cycle_blue_chip_dip_pooled",
        "cboe_putwrite_vol_regime_rotation",
        "closing_auction_deviation_reversal",
        "closing_auction_pressure_scalp",
        "commodity_backwardation_momentum_selection",
        "commodity_currency_leadlag",
        "commodity_open_interest_cycle_commodity_bond_overlay",
        "convertible_bond_moneyness_regime",
        "country_equity_etf_pooled_tsmom",
        "credit_spread_hy_ig_risk_regime",
        "cross_asset_divergence_fvg_continuation_daytrade",
        "cross_asset_financial_turbulence_derisking",
        "cross_asset_time_series_momentum",
        "crude_oil_term_structure_roll_carry",
        "customer_supplier_leadlag_semiconductors",
        "darvas_box_breakout_volume_confirmed",
        "defensive_sector_recession_signal_rotation",
        "dividend_growth_aristocrats",
        "dividend_month_premium_predicted_ex_month",
        "earnings_avwap_reaction_reclaim",
        "earnings_gap_continuation_intraday_confirm",
        "earnings_reaction_delayed_entry",
        "earnings_reaction_range_breakout_pead",
        "earnings_reaction_settle_drift",
        "em_dm_relative_strength_rotation",
        "equal_weight_rebalancing_premium",
        "etf_pairs_trading_stat_arb",
        "ex_dividend_day_capture_pooled",
        "faith_based_sin_stock_screen",
        "fallen_angel_bond_downgrade_drift",
        "fed_net_liquidity_regime",
        "fed_net_liquidity_regime_qqq",
        "fifty_two_week_high_momentum",
        "financials_yield_curve_steepening_rotation",
        "first_hour_range_breakout_swing",
        "fomc_post_statement_reversion_fade",
        "free_cash_flow_yield_cows",
        "fvg_reversal_cascade_continuation_daytrade",
        "fx_carry_distribution_yield_regime",
        "fx_dollar_index_trend_following",
        "fx_short_horizon_currency_momentum",
        "fx_value_ppp_mean_reversion",
        "gap_and_go_momentum_continuation",
        "gasoline_driving_season_seasonal",
        "gold_autumn_effect_seasonal",
        "gold_copper_ratio_fear_valuation_spy_overlay",
        "gold_long_treasury_relative_momentum_rotation",
        "gold_miners_real_rate_rotation",
        "green_day_morning_pullback_continuation_daytrade",
        "halloween_indicator_seasonal",
        "hedge_fund_13f_best_ideas",
        "high_dividend_yield_factor",
        "human_capital_employee_satisfaction",
        "hy_credit_bond_absolute_momentum",
        "insider_cluster_buying_value_overlay",
        "intl_govt_bond_time_series_momentum",
        "intraday_capitulation_reversal_bar",
        "intraday_half_hour_momentum_continuation",
        "intraday_liquidity_sweep_reversal",
        "intraday_rsi2_reversion_daytrade",
        "intraday_seasonality_continuation",
        "ipo_long_run_underperformance_fade",
        "ipo_long_run_underperformance_fade_beta_neutral",
        "january_small_cap_seasonal",
        "large_cap_single_stock_pooled_tsmom",
        "large_decline_rebound",
        "long_horizon_index_mean_reversion",
        "long_treasury_3x_fomc_window_drift",
        "long_treasury_auction_cycle_3x_inverse_rotation",
        "low_vol_high_beta_factor_momentum_rotation",
        "macro_release_jump_continuation",
        "managed_futures_trend_replication",
        "margin_debt_deleverage_synthetic_leverage_flip",
        "margin_debt_growth_regime",
        "merger_arbitrage_risk_arb",
        "money_market_fund_flow_riskoff",
        "money_market_fund_flow_riskoff_bounded_burnin",
        "month_end_rebalancing_flow_rotation",
        "monthly_opex_week_drift",
        "mortgage_reit_equity_reit_relative_strength_rotation",
        "move_vix_divergence_of_fear_gauges_derisking",
        "natural_gas_withdrawal_season_seasonal",
        "nr7_orb_trend_continuation",
        "oil_price_change_equity_underreaction",
        "options_pmcc_pooled_liquid",
        "options_put_ratio_spread_pooled_liquid",
        "options_wheel_pooled_liquid",
        "orb_5min_atr_stop_scalp",
        "overnight_gap_fade_intraday_reversal",
        "overnight_gap_meanreversion_fade",
        "overnight_intraday_tilt_pooled_largecap",
        "pead_earnings_surprise_drift",
        "post_opex_gamma_release_leverage_window",
        "pre_holiday_small_cap_effect",
        "preferred_stock_volatility_managed",
        "preferred_stock_yield_spread_reversion",
        "presidential_cycle_year3_seasonal",
        "quality_minus_junk_factor",
        "quarterly_index_rebalance_flow_leverage_window",
        "real_rate_gold_equity_rotation",
        "reit_broad_market_relative_strength_rotation",
        "reit_domestic_international_relative_strength_rotation",
        "reit_sector_absolute_momentum_trend",
        "reit_yield_spread_valuation_rotation",
        "residual_momentum_ff3proxy_top50_monthly",
        "retail_inventory_sales_breadth_short",
        "rsp_spy_breadth_ratio",
        "santa_claus_rally_yearend",
        "sector_ad_line_divergence_derisk",
        "sector_breadth_collapse_leverage_flip",
        "sector_breadth_thrust_zweig",
        "sector_composite_momentum_rotation",
        "sector_dispersion_regime_cyclical_stance",
        "sector_dual_momentum_cyclical_rotation",
        "sector_industry_momentum_formation",
        "sector_rrg_relative_strength_trend",
        "sector_same_month_seasonality",
        "selling_climax_capitulation_bounce_daytrade",
        "short_rate_easing_smallcap_rotation",
        "short_term_reversal_weekly",
        "single_day_extreme_decline_reversal_long",
        "size_premium_smallcap",
        "skew_index_tail_risk_regime",
        "small_cap_tsmom_long_inverse_rotation",
        "sp500_index_addition_effect",
        "spinoff_outperformance_effect",
        "stock_bond_correlation_regime_flip",
        "stock_split_post_split_drift_pooled",
        "synthetic_leveraged_dip_buy_pooled_largecap",
        "tips_nominal_breakeven_rotation",
        "treasury_duration_volatility_timing",
        "treasury_month_end_index_demand_overlay",
        "triple_witching_week_effect",
        "turtle_soup_failed_breakout",
        "turtle_soup_failed_breakout_reversal_daytrade",
        "us_industry_etf_pooled_tsmom",
        "us_leads_foreign_equity_diffusion",
        "usd_strength_risk_regime",
        "utilities_beta_rotation_relative_strength",
        "value_factor_book_to_market",
        "variance_risk_premium_harvest",
        "vcp_breakout_volume_confirmation_swing",
        "vix_spike_reversal_equity_buy",
        "vix_term_structure_carry",
        "vol_managed_intl_country_etf_pooled",
        "vol_managed_sector_spdr_pooled",
        "vol_managed_smallcap_leverage_switch",
        "vol_managed_spread_credit_pooled",
        "volume_filtered_orb_stocks_in_play_pooled",
        "vvix_vix_derisking_regime",
        "vwap_cross_decaying_conviction_exit_daytrade",
        "vwap_reclaim_reversal",
        "vwap_trend_intraday_scalp",
        "vxn_vix_spread_derisking_regime",
        "weekend_monday_effect",
        "wide_moat_quality_investing",
        "yield_curve_slope_duration_rotation",
    }


def test_load_catalog_rejects_missing_required_field(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"family_id": "x"}]))
    with pytest.raises(ValueError, match="missing required field"):
        catalog_mod.load_catalog(str(path))


def test_load_catalog_rejects_invalid_status(tmp_path):
    entry = _valid_entry(status="not_a_real_status")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([entry]))
    with pytest.raises(ValueError, match="invalid status"):
        catalog_mod.load_catalog(str(path))


def test_eligible_entries_filters_on_status():
    catalog = [_valid_entry(status="eligible", family_id="a"),
               _valid_entry(status="ruled_out", family_id="b")]
    assert [e["family_id"] for e in catalog_mod.eligible_entries(catalog)] == ["a"]
