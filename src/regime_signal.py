"""
regime_signal.py
----------------
Regime-gated leveraged-ETF allocation signal.

Reports whether a 3x position in TQQQ / SOXL / SPXL is justified under this
rule on the latest bar, by combining two research ingredients:

  1. A 2-state GaussianHMM **regime read** (bull/bear) on the UNDERLYING index
     (QQQ / ^SOX / SPY), using the fitting recipe in ``fit_hmm`` below.
  2. The project's core **leverage condition  mu > 2 sigma^2**  (3x geometrically
     out-compounds 1x only when the arithmetic mean return exceeds twice the
     variance), computed on a TRAILING window so it carries no lookahead.

Rule: **hold the 3x ETF only when CURRENT regime == bull AND trailing mu > 2 sigma^2.**
Otherwise rotate to the 1x underlying / cash.

LOOKAHEAD NOTE. Fitting one HMM on the full sample and reading *historical*
labels embeds lookahead, so those labels are NOT tradeable.
This module reads ONLY the *last* observation's filtered state from a fit on all
data through today. At the final timestep the forward-backward posterior equals
the filtered (forward-only) posterior — no future data exists to leak — so the
*current-day* read is legitimately tradeable. (A production deployment refits
daily on the expanding window; the HMM parameters are stable enough that the
single fit here is a faithful stand-in for the current read.)

This module is price-data only and connects to no brokerage account. A caller
may optionally pass a `holdings` dict to `build_signal` to get a
position-aware action label.
"""

import os
import logging
import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM

# hmmlearn logs a benign "Model is not converging" line per restart when EM hits
# its tolerance plateau (the best-of-N-restarts model is still selected by score).
# Quiet it so the signal output stays readable.
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

TRADING_DAYS = 252
RANDOM_STATE = 42

# 3x leveraged ETF  ->  underlying index series used for the regime read.
LEV_MAP = {"TQQQ": "QQQ", "SOXL": "SOX", "SPXL": "SPY"}
# 3x leveraged ETF  ->  cost-adjusted synthetic 3x series (for the historical
# regime-conditional edge check; extends back to the underlying's full history).
SYN_MAP = {"TQQQ": "QQQ_3x_synthetic", "SOXL": "SOX_3x_synthetic",
           "SPXL": "SPY_3x_synthetic"}


# ── Data ────────────────────────────────────────────────────────────────────────

def load_series(ticker, data_dir):
    """Load data/raw/<ticker>.csv -> DataFrame[Date, <ticker>], sorted."""
    fp = os.path.join(data_dir, f"{ticker}.csv")
    df = pd.read_csv(fp, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


# ── HMM fitting recipe ────────────────────────────────────────────────────────────

def fit_hmm(prices_df, ticker, n_components=2, n_restarts=15):
    """
    Fit a 2-state GaussianHMM on log returns, relabeled so state 0 = BEAR
    (lower daily mean) and state 1 = BULL.
    Returns (model, log_return_series, state_array, dates).
    """
    df = prices_df.copy()
    df["lr"] = np.log1p(df[ticker].pct_change())
    df = df.dropna(subset=["lr"]).reset_index(drop=True)
    r = df["lr"]
    dates = pd.to_datetime(df["Date"])
    X = r.values.reshape(-1, 1)

    best_model, best_score = None, -np.inf
    for rs in range(n_restarts):
        m = GaussianHMM(n_components=n_components, covariance_type="full",
                        n_iter=1000, random_state=RANDOM_STATE + rs)
        try:
            m.fit(X)
            score = m.score(X)
            if score > best_score:
                best_score, best_model = score, m
        except Exception:
            pass

    states = best_model.predict(X)
    # Relabel so state 0 = bear (lower mean return).
    if best_model.means_[0, 0] > best_model.means_[1, 0]:
        states = 1 - states
        best_model.means_ = best_model.means_[[1, 0]]
        best_model.covars_ = best_model.covars_[[1, 0]]
        best_model.transmat_ = best_model.transmat_[[1, 0]][:, [1, 0]]
        best_model.startprob_ = best_model.startprob_[[1, 0]]

    return best_model, r, states, dates


def _mu_over_2sigma2(log_returns):
    """mu / (2 sigma^2) on an annualized basis. >1 => 3x geometrically justified."""
    mu = log_returns.mean() * TRADING_DAYS
    s2 = log_returns.var() * TRADING_DAYS
    ratio = mu / (2 * s2) if s2 > 1e-12 else np.nan
    return float(mu), float(s2), float(ratio)


# ── Current regime read ──────────────────────────────────────────────────────────

def current_regime(prices_df, ticker, vol_window=63, return_model=False,
                   n_restarts=15):
    """
    Tradeable current-day regime read for one underlying.

    Returns a dict:
      asof, last_price, regime ('bull'/'bear'), p_bull (filtered posterior),
      regime_mu_ann, regime_2sig2_ann, regime_ratio   (HMM regime-conditional),
      trail_mu_ann, trail_2sig2_ann, trail_ratio       (trailing vol_window),
      leverage_justified (regime==bull AND trail_ratio>1).

    return_model: if True, returns (info, model, states) instead of just
      info -- lets the live-signal guard (hmm_guards.fit_is_sane) inspect the
      fitted model and state path directly.
    n_restarts: number of random EM restarts (default 15; the best-scoring
      fit across restarts is kept).
    """
    model, r, states, dates = fit_hmm(prices_df, ticker, n_restarts=n_restarts)
    X = r.values.reshape(-1, 1)
    p_bull = float(model.predict_proba(X)[-1, 1])     # filtered at the last obs
    cur = int(states[-1])
    regime = "bull" if cur == 1 else "bear"

    # HMM regime-conditional stats for the CURRENT regime.
    mask = states == cur
    reg_mu, reg_sig2, reg_ratio = _mu_over_2sigma2(r[mask])

    # Trailing (no-lookahead) mu>2sigma^2 over the last vol_window days.
    tail = r.iloc[-vol_window:]
    tr_mu = float(tail.mean() * TRADING_DAYS)
    tr_sig2 = float(tail.var() * TRADING_DAYS)
    tr_ratio = tr_mu / (2 * tr_sig2) if tr_sig2 > 1e-12 else np.nan

    leverage_justified = (regime == "bull") and (tr_ratio > 1.0)

    info = {
        "asof": dates.iloc[-1].date().isoformat(),
        "last_price": float(prices_df[ticker].iloc[-1]),
        "regime": regime,
        "p_bull": p_bull,
        "regime_mu_ann": reg_mu,
        "regime_sigma2_ann": reg_sig2,
        "regime_ratio": float(reg_ratio),
        "trail_mu_ann": tr_mu,
        "trail_sigma2_ann": tr_sig2,
        "trail_ratio": float(tr_ratio),
        "vol_window": vol_window,
        "leverage_justified": bool(leverage_justified),
    }
    if return_model:
        return info, model, states
    return info


def regime_conditional_edge(data_dir, lev_etf):
    """
    Historical justification for the gate: realized 3x-vs-1x out-compounding,
    split by HMM regime, using the cost-adjusted synthetic 3x series. Returns a
    dict of per-regime annualized log-CAGR for 1x and 3x. (Full-sample HMM here —
    this is a descriptive backtest, not the tradeable current read.)
    """
    und = LEV_MAP[lev_etf]
    syn = SYN_MAP[lev_etf]
    df1 = load_series(und, data_dir)
    df3 = load_series(syn, data_dir)
    _, r1, states, dates = fit_hmm(df1, und)
    # align synthetic 3x log returns to the same dated index
    s3 = df3.set_index("Date")[syn]
    lr3 = np.log1p(s3.pct_change())
    idx = dates.values
    out = {}
    for st, lab in [(1, "bull"), (0, "bear")]:
        mask = states == st
        d_sel = dates[mask]
        r1_sel = r1[mask]
        r3_sel = lr3.reindex(d_sel).dropna()
        out[lab] = {
            "n_days": int(mask.sum()),
            "cagr_1x": float(np.expm1(r1_sel.mean() * TRADING_DAYS)),
            "cagr_3x": float(np.expm1(r3_sel.mean() * TRADING_DAYS)) if len(r3_sel) else np.nan,
        }
    return out


# ── Recommendation ───────────────────────────────────────────────────────────────

def recommend(info, holding_3x=None):
    """
    Map a current_regime() dict + (optional) current holding to an action.

    holding_3x: True if a position in the 3x ETF is held, False if not,
    None if unknown -> returns the gate state only.
    """
    risk_on = info["leverage_justified"]
    gate = "RISK-ON  (3x justified)" if risk_on else "RISK-OFF (rotate to 1x/cash)"
    if holding_3x is None:
        action = "ENTER/HOLD 3x" if risk_on else "EXIT/AVOID 3x"
    elif holding_3x and risk_on:
        action = "HOLD 3x"
    elif holding_3x and not risk_on:
        action = "EXIT 3x -> 1x/cash"
    elif (not holding_3x) and risk_on:
        action = "ENTER 3x"
    else:
        action = "STAY OUT"
    return {"gate": gate, "risk_on": bool(risk_on), "action": action}


def build_signal(data_dir, holdings=None, vol_window=63):
    """
    Full regime-gated signal table across TQQQ / SOXL / SPXL.

    holdings: optional caller-supplied dict {lev_etf: bool} of held positions.
    Missing keys -> holding unknown (None).
    Returns a tidy DataFrame, one row per 3x ETF.
    """
    holdings = holdings or {}
    rows = []
    for lev, und in LEV_MAP.items():
        info = current_regime(load_series(und, data_dir), und, vol_window)
        rec = recommend(info, holdings.get(lev))
        rows.append({
            "lev_etf": lev, "underlying": und, "asof": info["asof"],
            "regime": info["regime"], "p_bull": round(info["p_bull"], 3),
            "trail_mu_ann": round(info["trail_mu_ann"], 4),
            "trail_2sigma2_ann": round(2 * info["trail_sigma2_ann"], 4),
            "trail_ratio": round(info["trail_ratio"], 2),
            "regime_ratio": round(info["regime_ratio"], 2),
            "leverage_justified": info["leverage_justified"],
            "holding": holdings.get(lev),
            "gate": rec["gate"], "action": rec["action"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    DATA = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    sig = build_signal(DATA)
    pd.set_option("display.width", 160, "display.max_columns", 20)
    print(sig.to_string(index=False))
