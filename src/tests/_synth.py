"""Tiny synthetic two-regime series for the gate-sweep test suite.

Prices are built as P_t = 100 * exp(cumsum(r)), so log1p(pct_change) of the
price series recovers r exactly -- fit_hmm sees exactly the returns we drew.
"""
import numpy as np
import pandas as pd


def two_regime_returns(n_bull=400, n_bear=150, n_bull2=200, seed=7):
    """Calm-up / volatile-down / calm-up daily log returns.

    The bear mean is -0.008 (not a realistic -0.002) ON PURPOSE: with sd=0.03
    and n=150 the bear sample mean has SE ~0.0025, so a -0.002 true mean makes
    the FITTED state-mean ordering pure seed luck (some seeds give a mean gap
    so small that fit_is_sane correctly rejects the fit). At -0.008
    the separation is >3 SE, so the HMM's relabel-by-mean is stable across
    seeds and guard-sane. Do not weaken this for realism -- these fixtures
    test label/guard mechanics, not market realism.
    """
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.normal(0.0008, 0.010, n_bull),
        rng.normal(-0.0080, 0.030, n_bear),
        rng.normal(0.0008, 0.010, n_bull2),
    ])


def prices_from_returns(r, ticker="PX", start="2015-01-02", p0=100.0):
    """DataFrame[Date, <ticker>] whose log returns equal r exactly."""
    dates = pd.bdate_range(start, periods=len(r) + 1)
    prices = p0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    return pd.DataFrame({"Date": dates.date, ticker: prices})
