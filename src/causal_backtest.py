"""Causal walk-forward machinery for the gate-parameter sweep.

Produces the CAUSAL daily p_bull path a live trader could actually have
computed: refit every `refit_every` bars (schedule anchored to ABSOLUTE bar
index so truncation invariance holds), forward-filter between refits with
the params frozen at the last refit. Runs under an interpreter with the requirements installed (hmmlearn via
regime_signal). Reads ONLY data/raw corpus CSVs -- never a broker feed or any
live data source.
"""
import json
import os

import numpy as np
import pandas as pd

import hmm_guards
from regime_signal import fit_hmm, load_series, LEV_MAP, SYN_MAP

TRADING_DAYS = 252


def forward_filter_pbull_path(params, log_returns):
    """Filtered P(bull | r_0..r_t) for EVERY t, one scaled forward pass.

    Same recursion as hmm_guards.forward_filter_pbull (kept frozen there);
    the per-step emission makes prefix-consistency an algebraic identity:
    alpha at step t depends only on r_0..r_t.
    """
    means = np.asarray(params["means"], dtype=float)
    variances = np.asarray(params["variances"], dtype=float)
    transmat = np.asarray(params["transmat"], dtype=float)
    startprob = np.asarray(params["startprob"], dtype=float)
    x = np.asarray(log_returns, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("forward_filter_pbull_path: empty return series")

    def _pdf(xt):
        return np.exp(-0.5 * (xt - means) ** 2 / variances) / np.sqrt(
            2.0 * np.pi * variances)

    out = np.empty(x.size, dtype=float)
    alpha = startprob * _pdf(x[0])
    s = alpha.sum()
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("forward_filter_pbull_path: invalid emission at t=0")
    alpha /= s
    out[0] = alpha[1]
    for t in range(1, x.size):
        alpha = (alpha @ transmat) * _pdf(x[t])
        s = alpha.sum()
        if not np.isfinite(s) or s <= 0.0:
            raise ValueError(
                "forward_filter_pbull_path: underflow/invalid emission")
        alpha /= s
        out[t] = alpha[1]
    return out


def trailing_ratio_path(prices_df, ticker, vol_window=63):
    """Rolling trailing mu/(2 sigma^2), annualized, ddof=1 -- the exact math
    of regime_signal.current_regime's tail block, at every bar."""
    lr = np.log1p(prices_df[ticker].pct_change()).dropna().reset_index(drop=True)
    mu = lr.rolling(vol_window).mean() * TRADING_DAYS
    sig2 = lr.rolling(vol_window).var() * TRADING_DAYS       # ddof=1 default
    ratio = mu / (2.0 * sig2)
    ratio[sig2 <= 1e-12] = np.nan
    return ratio


def causal_pbull_path(prices_df, ticker, refit_every=21, warmup=1000,
                      n_restarts=15):
    """CAUSAL daily p_bull: refit on bars [0..k) at every absolute bar index
    k in {warmup, warmup+refit_every, ...}; between refits, forward-filter
    r[0..t] with the last SANE params. Bar t's value uses data through t and
    params through the last refit <= t -- nothing from the future.
    """
    df = prices_df.copy()
    lr_all = np.log1p(df[ticker].pct_change()).dropna()
    dates = pd.to_datetime(df["Date"]).iloc[1:].dt.date.reset_index(drop=True)
    x = lr_all.values
    T = x.size
    if T <= warmup:
        return pd.DataFrame(columns=["Date", "p_bull", "refit_bar"])

    rows = []
    params, params_bar = None, None
    for k in range(warmup, T, refit_every):
        seg_end = min(k + refit_every, T)
        # refit on everything strictly before bar k
        sub = df.iloc[:k + 1]                  # k+1 price rows -> k lr bars
        try:
            model, _r, states, _d = fit_hmm(sub, ticker,
                                            n_restarts=n_restarts)
            ok, _reasons = hmm_guards.fit_is_sane(model, states)
        except Exception:
            ok = False
        if ok:
            params = hmm_guards.params_to_dict(model)
            params_bar = k
        if params is None:
            continue                            # no sane fit yet: skip segment
        seg_path = forward_filter_pbull_path(params, x[:seg_end])[k:seg_end]
        for i, p in enumerate(seg_path):
            rows.append((dates.iloc[k + i], float(p), params_bar))
    return pd.DataFrame(rows, columns=["Date", "p_bull", "refit_bar"])


def _safe_forward_path(params, log_returns):
    """Scaled forward pass (same recursion as forward_filter_pbull_path) that,
    on emission underflow at step j, returns the VALID prefix out[:j] and j
    instead of raising. fail_idx == len(x) when the whole pass succeeds."""
    means = np.asarray(params["means"], dtype=float)
    variances = np.asarray(params["variances"], dtype=float)
    transmat = np.asarray(params["transmat"], dtype=float)
    startprob = np.asarray(params["startprob"], dtype=float)
    x = np.asarray(log_returns, dtype=float).ravel()
    n = x.size
    out = np.empty(n, dtype=float)
    if n == 0:
        return out[:0], 0

    def _pdf(xt):
        return np.exp(-0.5 * (xt - means) ** 2 / variances) / np.sqrt(
            2.0 * np.pi * variances)

    alpha = startprob * _pdf(x[0])
    s = alpha.sum()
    if not np.isfinite(s) or s <= 0.0:
        return out[:0], 0
    alpha /= s
    out[0] = alpha[1]
    for t in range(1, n):
        alpha = (alpha @ transmat) * _pdf(x[t])
        s = alpha.sum()
        if not np.isfinite(s) or s <= 0.0:
            return out[:t], t
        alpha /= s
        out[t] = alpha[1]
    return out, n


def causal_pbull_path_faultsoft(prices_df, ticker, refit_every=21, warmup=1000,
                                n_restarts=15):
    """ADDITIVE bar-faithful fault-soft twin of causal_pbull_path.
    Identical p_bull on every successfully-computed bar; on the first underflow
    bar of a segment, that bar and the segment remainder HOLD the last sane
    p_bull with degraded=True, while earlier bars in the segment keep their
    correct values. causal_pbull_path stays byte-identical -> ABI/sweep frames
    unchanged (guarded by the golden-source test)."""
    df = prices_df.copy()
    lr_all = np.log1p(df[ticker].pct_change()).dropna()
    dates = pd.to_datetime(df["Date"]).iloc[1:].dt.date.reset_index(drop=True)
    x = lr_all.values
    T = x.size
    if T <= warmup:
        return pd.DataFrame(columns=["Date", "p_bull", "refit_bar", "degraded"])

    rows = []
    params, params_bar = None, None
    last_p = None
    for k in range(warmup, T, refit_every):
        seg_end = min(k + refit_every, T)
        sub = df.iloc[:k + 1]
        try:
            model, _r, states, _d = fit_hmm(sub, ticker, n_restarts=n_restarts)
            ok, _reasons = hmm_guards.fit_is_sane(model, states)
        except Exception:
            ok = False
        if ok:
            params = hmm_guards.params_to_dict(model)
            params_bar = k
        if params is None:
            continue
        path, fail_idx = _safe_forward_path(params, x[:seg_end])
        for i in range(k, seg_end):
            if i < fail_idx:
                last_p = float(path[i])
                rows.append((dates.iloc[i], last_p, params_bar, False))
            else:
                held = last_p if last_p is not None else 0.5
                rows.append((dates.iloc[i], float(held), params_bar, True))
    return pd.DataFrame(rows, columns=["Date", "p_bull", "refit_bar", "degraded"])


UND_TO_SYN = {v: SYN_MAP[k] for k, v in LEV_MAP.items()}
# Dow wiring: a LOCAL entry, deliberately NOT added to
# regime_signal.LEV_MAP/SYN_MAP -- those drive the regime signal's
# TQQQ/SOXL/SPXL legs, which stay byte-identical. DIA is a sweep-only
# research ticker.
UND_TO_SYN["DIA"] = "DIA_3x_synthetic"
# Deep index-owner corpora: same LOCAL-only pattern as DIA.
# regime_signal.LEV_MAP/SYN_MAP stay byte-identical -- the regime signal
# keeps its TQQQ/SOXL/SPXL legs on the QQQ/SOX/SPY corpora. These are
# research corpora sourced from the index owner; see src/deep_corpus.py.
UND_TO_SYN["NDX_TR"] = "NDX_TR_3x_synthetic"
UND_TO_SYN["SPX_TR"] = "SPX_TR_3x_synthetic"


def _log_returns_by_date(df, col):
    lr = np.log1p(df[col].pct_change())
    dates = pd.to_datetime(df["Date"]).dt.date
    return pd.Series(lr.values, index=dates.values).iloc[1:]


def _file_sha1(path):
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


CASH_TRANSFORM_VERSION = 1


def cash_log_returns_by_date(frame_dates, data_dir):
    """Daily cash log-return per frame date, using the ^IRX rate known STRICTLY
    BEFORE that date (no same-bar rate lookahead), forward-filled (never
    backfilled)."""
    irx = pd.read_csv(os.path.join(data_dir, "IRX.csv"), parse_dates=["Date"])
    irx = irx.dropna().drop_duplicates("Date").sort_values("Date")
    rate = pd.Series(irx["irx_rate_annual"].values,
                     index=pd.to_datetime(irx["Date"]).values)
    fd = pd.to_datetime(list(frame_dates))
    prior = rate.reindex(rate.index.union(fd)).ffill().shift(1).reindex(fd)
    daily = np.log1p(prior.values.astype(float) / TRADING_DAYS)
    return pd.Series(daily, index=pd.to_datetime(fd).date)


def build_joint_frame(ticker, refit_every, data_dir, cache_dir,
                      vol_window=63, warmup=1000, n_restarts=15,
                      force=False, strict=False):
    """[Date, p_bull, trail_ratio, lr_1x, lr_3x, lr_cash] for one underlying,
    cached.

    NaN trail_ratio rows are KEPT: the gate engine treats NaN as disarm,
    exactly like live apply_hysteresis.
    """
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.join(cache_dir, f"{ticker}_re{refit_every}")
    csv_fp, meta_fp = base + ".csv", base + ".meta.json"

    und = load_series(ticker, data_dir)
    syn = load_series(UND_TO_SYN[ticker], data_dir)
    # content fingerprints: corpus_max_date alone is blind to a regenerated
    # source CSV whose last date did not change (e.g. a regenerated input)
    meta = {"corpus_max_date": str(pd.to_datetime(und["Date"]).max().date()),
            "vol_window": vol_window, "warmup": warmup,
            "n_restarts": n_restarts,
            "und_sha1": _file_sha1(
                os.path.join(data_dir, f"{ticker}.csv")),
            "syn_sha1": _file_sha1(
                os.path.join(data_dir, f"{UND_TO_SYN[ticker]}.csv"))}
    meta["irx_sha1"] = _file_sha1(os.path.join(data_dir, "IRX.csv"))
    meta["cash_transform_version"] = CASH_TRANSFORM_VERSION

    if not force and os.path.exists(csv_fp) and os.path.exists(meta_fp):
        try:
            with open(meta_fp) as fh:
                cached_meta = json.load(fh)
            if cached_meta == meta:
                out = pd.read_csv(csv_fp, parse_dates=["Date"],
                                  float_precision="round_trip")
                out["Date"] = out["Date"].dt.date
                if (list(out.columns) == ["Date", "p_bull", "trail_ratio",
                                          "lr_1x", "lr_3x", "lr_cash"]
                        and len(out) > 0):
                    return out
                # schema-broken or empty cache -> fall through to rebuild
        except Exception:
            pass          # corrupt/unreadable cache -> rebuild (fail-safe)

    if strict:
        raise FileNotFoundError(
            f"strict=True: no valid causal cache for {ticker} re{refit_every} "
            f"in {cache_dir} (meta mismatch or absent); refusing to refit.")

    path = causal_pbull_path(und, ticker, refit_every=refit_every,
                             warmup=warmup, n_restarts=n_restarts)
    ratio = trailing_ratio_path(und, ticker, vol_window=vol_window)
    lr1 = _log_returns_by_date(und, ticker)
    dates1 = pd.to_datetime(und["Date"]).dt.date.iloc[1:]
    ratio_by_date = pd.Series(ratio.values, index=dates1.values)
    lr3 = _log_returns_by_date(syn, UND_TO_SYN[ticker])

    out = path[["Date", "p_bull"]].copy()
    out["trail_ratio"] = out["Date"].map(ratio_by_date)
    out["lr_1x"] = out["Date"].map(lr1)
    out["lr_3x"] = out["Date"].map(lr3)
    out["lr_cash"] = cash_log_returns_by_date(out["Date"].values, data_dir).values
    out["lr_cash"] = out["lr_cash"].fillna(0.0)
    out = out.dropna(subset=["lr_1x", "lr_3x"]).reset_index(drop=True)

    tmp = csv_fp + ".tmp"
    out.to_csv(tmp, index=False)
    os.replace(tmp, csv_fp)
    with open(meta_fp, "w") as fh:
        json.dump(meta, fh)
    return out


def causal_pbull_path_from_params(prices_df, ticker, boundary_params,
                                  refit_every=21, warmup=1000):
    """ADDITIVE replay seam: rerun the exact runtime path of
    causal_pbull_path_faultsoft from ARCHIVED per-boundary params -- no fitting,
    ever. boundary_params: {absolute boundary LR-bar k -> params dict or None};
    None = that boundary's production fit failed/was rejected, and the replay
    retains the nearest earlier non-None params, exactly as the production engine retained
    them. ALWAYS call with the FULL frozen series -- a row-filtered frame
    shifts the absolute-bar grid and merges sessions' returns.
    Output contract identical to causal_pbull_path_faultsoft.
    """
    df = prices_df.copy()
    lr_all = np.log1p(df[ticker].pct_change()).dropna()
    dates = pd.to_datetime(df["Date"]).iloc[1:].dt.date.reset_index(drop=True)
    x = lr_all.values
    T = x.size
    if T <= warmup:
        return pd.DataFrame(columns=["Date", "p_bull", "refit_bar", "degraded"])

    rows = []
    params, params_bar = None, None
    last_p = None
    for k in range(warmup, T, refit_every):
        seg_end = min(k + refit_every, T)
        archived = boundary_params.get(k)
        if archived is not None:
            params, params_bar = archived, k
        if params is None:
            continue
        path, fail_idx = _safe_forward_path(params, x[:seg_end])
        for i in range(k, seg_end):
            if i < fail_idx:
                last_p = float(path[i])
                rows.append((dates.iloc[i], last_p, params_bar, False))
            else:
                held = last_p if last_p is not None else 0.5
                rows.append((dates.iloc[i], float(held), params_bar, True))
    return pd.DataFrame(rows, columns=["Date", "p_bull", "refit_bar", "degraded"])
