"""Noise-aware statistics.

Everything an hourly signal produces is autocorrelated (overlapping targets, smoothed
positions), so iid standard errors understate uncertainty. We use:

* Newey-West (Bartlett) long-run variance for SEs of means.
* Sharpe SE: sqrt( (LRV/Var) * (1 + SR^2/2) / T )   -- Lo (2002) scaled by the variance ratio.
* Politis-Romano stationary bootstrap for confidence intervals.
* Effective sample size T_eff = T * Var / LRV for the probabilistic / deflated Sharpe.
* Deflated Sharpe ratio (Bailey & Lopez de Prado 2014) for the number of trials.
* Normal-normal shrinkage of the Sharpe toward 0 with prior sd tau.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from lab.backtest import align, flat_stats, pnl, sharpe

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329


# ----------------------------------------------------------------------------- basics
def nw_lrv(x, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 3:
        return float("nan")
    x = x - x.mean()
    g0 = float(x @ x) / T
    s = g0
    for k in range(1, min(lag, T - 1) + 1):
        g = float(x[k:] @ x[: T - k]) / T
        s += 2.0 * (1.0 - k / (lag + 1.0)) * g
    return max(s, 1e-12 * g0) if g0 > 0 else 0.0


def mean_se_nw(x, lag: int) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan"), float("nan")
    return float(x.mean()), math.sqrt(nw_lrv(x, lag) / len(x))


def sharpe_se_nw(r, lag: int) -> tuple[float, float, float]:
    """(SR, SE_nw, variance ratio LRV/Var)."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    sr = sharpe(r)
    if not np.isfinite(sr):
        return sr, float("nan"), float("nan")
    var = r.var(ddof=1)
    vr = nw_lrv(r, lag) / var if var > 0 else float("nan")
    se = math.sqrt(max(vr, 1e-12) * (1.0 + 0.5 * sr * sr) / len(r))
    return sr, se, vr


def moments(r) -> tuple[float, float]:
    """(skew, non-excess kurtosis)."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 4:
        return 0.0, 3.0
    z = (r - r.mean()) / r.std()
    return float((z ** 3).mean()), float((z ** 4).mean())


# ----------------------------------------------------------------------------- bootstrap
def stationary_bootstrap_index(T: int, block: float, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = 1.0 / max(block, 1.0)
    idx = np.empty((reps, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, reps)
    jumps = rng.random((reps, T)) < p
    starts = rng.integers(0, T, (reps, T))
    for t in range(1, T):
        idx[:, t] = np.where(jumps[:, t], starts[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def bootstrap_stat(x, stat: str, block: float, reps: int, seed: int, alpha: float = 0.10) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return {"lo": float("nan"), "hi": float("nan"), "se": float("nan")}
    idx = stationary_bootstrap_index(len(x), block, reps, seed)
    xs = x[idx]
    if stat == "mean":
        vals = xs.mean(axis=1)
    elif stat == "sharpe":
        sd = xs.std(axis=1, ddof=1)
        vals = np.where(sd > 0, xs.mean(axis=1) / np.where(sd > 0, sd, 1.0), np.nan)
    else:
        raise ValueError(stat)
    vals = vals[np.isfinite(vals)]
    return {
        "lo": float(np.quantile(vals, alpha / 2)),
        "hi": float(np.quantile(vals, 1 - alpha / 2)),
        "se": float(vals.std(ddof=1)),
    }


# ----------------------------------------------------------------------------- inference
def expected_max_sharpe(n_trials: int, sr_sd: float) -> float:
    """E[max of n iid N(0, sr_sd^2)] (Bailey & Lopez de Prado approximation)."""
    if n_trials <= 1 or not np.isfinite(sr_sd):
        return 0.0
    z1 = _N.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _N.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return sr_sd * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def probabilistic_sharpe(sr: float, sr_ref: float, t_eff: float, skew: float, kurt: float) -> float:
    if not (np.isfinite(sr) and t_eff > 2):
        return float("nan")
    den = math.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr, 1e-12))
    return _N.cdf((sr - sr_ref) * math.sqrt(t_eff - 1.0) / den)


def deflated_sharpe(sr, t_eff, skew, kurt, n_trials, sr_sd_trials) -> tuple[float, float]:
    """(SR* benchmark, DSR = P[true SR > SR*])."""
    sr_star = expected_max_sharpe(n_trials, sr_sd_trials)
    return sr_star, probabilistic_sharpe(sr, sr_star, t_eff, skew, kurt)


def shrink_sharpe(sr: float, se: float, tau: float) -> dict:
    """Posterior of true SR under prior N(0, tau^2) and likelihood N(sr, se^2)."""
    if not (np.isfinite(sr) and np.isfinite(se) and se > 0):
        return {"post_mean": float("nan"), "post_sd": float("nan"), "p_positive": float("nan")}
    w = tau * tau / (tau * tau + se * se)
    m = w * sr
    sd = math.sqrt(w) * se
    return {"post_mean": m, "post_sd": sd, "p_positive": _N.cdf(m / sd)}


def benjamini_hochberg(pvals, q: float = 0.10) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    k = np.max(np.flatnonzero(passed)) + 1 if passed.any() else 0
    out = np.zeros(n, dtype=bool)
    out[order[:k]] = True
    return out


def p_two_sided(t: float) -> float:
    return 2.0 * (1.0 - _N.cdf(abs(t))) if np.isfinite(t) else float("nan")


# ----------------------------------------------------------------------------- signal geometry
def row_cos(A: pd.DataFrame, B: pd.DataFrame) -> pd.Series:
    """Per-row cosine of de-meaned vectors (NaN when either row is all zero)."""
    cols = [c for c in A.columns if c in B.columns]
    idx = A.index.intersection(B.index)
    a = A.loc[idx, cols].to_numpy(float)
    b = B.loc[idx, cols].to_numpy(float)
    a = np.nan_to_num(a - np.nanmean(a, axis=1, keepdims=True))
    b = np.nan_to_num(b - np.nanmean(b, axis=1, keepdims=True))
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (a * b).sum(axis=1) / (na * nb)
    c[(na == 0) | (nb == 0)] = np.nan
    return pd.Series(c, index=idx)


def ic_series(P: pd.DataFrame, T: pd.DataFrame, lag: int = 1) -> pd.Series:
    """rho(i) = cos(P(i-lag), T(i)); rows where the target is degenerate are dropped."""
    s = row_cos(P.shift(lag), T)
    return s.dropna()


def ic_decay(P: pd.DataFrame, T: pd.DataFrame, lags=(1, 2, 4, 8, 16, 24, 48, 96)) -> dict:
    return {int(k): float(ic_series(P, T, k).mean()) for k in lags}


def concentration_raw(P: pd.DataFrame) -> float:
    """||mean of unit rows|| in the TICKER frame of one period (proxy only; the official
    number uses the gauge frame, see lab/gauge.py)."""
    A = P.fillna(0.0).to_numpy(float)
    A = A - A.mean(axis=1, keepdims=True)
    n = np.linalg.norm(A, axis=1)
    U = A[n > 0] / n[n > 0, None]
    return float(np.linalg.norm(U.mean(axis=0))) if len(U) else float("nan")


# ----------------------------------------------------------------------------- full summary
def monthly_table(net: pd.Series, months: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"r": net, "m": months.reindex(net.index)})
    g = df.groupby("m")["r"]
    return pd.DataFrame({"sharpe": g.apply(sharpe), "mean": g.mean(), "hours": g.size()})


def summarize(P: pd.DataFrame, panel, s: dict, n_trials: int = 1, sr_sd_trials: float | None = None) -> dict:
    """All headline metrics, each with its uncertainty."""
    from lab.io import month_labels

    lag = int(s["nw_lag_hours"])
    blk = float(s["bootstrap_block_hours"])
    reps = int(s["bootstrap_reps"])
    seed = int(s["seed"])
    out: dict = {"hours": int(len(P))}

    by_cost = {}
    for c in s["costs_bp"]:
        pl = pnl(P, panel.returns, c)
        sr, se, vr = sharpe_se_nw(pl["net"], lag)
        ci = bootstrap_stat(pl["net"].to_numpy(), "sharpe", blk, reps, seed)
        by_cost[c] = (pl, sr, se, vr, ci)
        tag = f"{c:g}bp"
        out[f"SR_net_{tag}"] = sr
        out[f"SR_se_{tag}"] = se
        out[f"SR_ci90_lo_{tag}"] = ci["lo"]
        out[f"SR_ci90_hi_{tag}"] = ci["hi"]

    c0 = float(s["official_cost_bp"])
    pl, sr, se, vr, _ = by_cost[c0]
    out["SR_gross"] = sharpe(pl["gross"])
    out["variance_ratio"] = vr
    out["turnover_mean"] = float(pl["turnover"].mean())
    gsd = pl["gross"].std()
    out["cost_drag_sd_units"] = float(pl["cost"].mean() / gsd) if gsd > 0 else float("nan")

    skew, kurt = moments(pl["net"])
    t_eff = len(pl) / vr if np.isfinite(vr) and vr > 0 else len(pl)
    out.update({"skew": skew, "kurt": kurt, "T_eff": float(t_eff)})
    out["PSR0"] = probabilistic_sharpe(sr, 0.0, t_eff, skew, kurt)
    sd_trials = sr_sd_trials if (sr_sd_trials and np.isfinite(sr_sd_trials)) else se
    out["SR_star"], out["DSR"] = deflated_sharpe(sr, t_eff, skew, kurt, n_trials, sd_trials)
    out["n_trials"] = int(n_trials)
    sh = shrink_sharpe(sr, se, float(s["prior_tau_sharpe"]))
    out["SR_shrunk"], out["SR_post_sd"], out["P_SR_pos"] = sh["post_mean"], sh["post_sd"], sh["p_positive"]

    months = panel.months()
    mt = monthly_table(pl["net"], months)
    out["months"] = int(len(mt))
    out["months_pos_frac"] = float((mt["sharpe"] > 0).mean()) if len(mt) else float("nan")
    if len(mt) > 2:
        mm = mt["mean"].to_numpy()
        out["months_t"] = float(mm.mean() / (mm.std(ddof=1) / math.sqrt(len(mm)))) if mm.std(ddof=1) > 0 else float("nan")
    out["monthly_sharpe"] = {k: float(v) for k, v in mt["sharpe"].items()}

    tl = int(s["target_lag"])
    rho = ic_series(P, panel.target, tl)
    ic, ic_se = mean_se_nw(rho.to_numpy(), lag)
    out.update({"IC": ic, "IC_se": ic_se, "IC_t": ic / ic_se if ic_se and ic_se > 0 else float("nan")})
    out["IC_decay"] = ic_decay(P, panel.target)
    if panel.dispersion is not None:
        d = panel.dispersion.reindex(rho.index)
        ok = d.notna() & (d > 0)
        out["IC_disp_corr"] = float(np.corrcoef(rho[ok], d[ok])[0, 1]) if ok.sum() > 10 else float("nan")

    out["concentration_raw"] = concentration_raw(P)
    out.update(flat_stats(P, float(s["flat"]["rel_threshold"])))
    return out


def summarize_folds(folds: list, s: dict, n_trials: int = 1, sr_sd_trials: float | None = None) -> dict:
    """Same metrics as `summarize`, for the official per-period protocol (lab.official). Each fold
    is scored on its own validation rows and starts flat; per-fold series are concatenated."""
    from lab.official import fold_pnl, stacked

    lag = int(s["nw_lag_hours"])
    blk = float(s["bootstrap_block_hours"])
    reps = int(s["bootstrap_reps"])
    seed = int(s["seed"])
    P_all = stacked(folds, "P")
    out: dict = {"hours": int(len(P_all)), "protocol": "official-per-period"}
    pls = {}
    for c in s["costs_bp"]:
        pl = fold_pnl(folds, c)
        pls[c] = pl
        sr, se, vr = sharpe_se_nw(pl["net"], lag)
        ci = bootstrap_stat(pl["net"].to_numpy(), "sharpe", blk, reps, seed)
        tag = f"{c:g}bp"
        out[f"SR_net_{tag}"], out[f"SR_se_{tag}"] = sr, se
        out[f"SR_ci90_lo_{tag}"], out[f"SR_ci90_hi_{tag}"] = ci["lo"], ci["hi"]
    c0 = float(s["official_cost_bp"])
    pl = pls[c0]
    sr, se, vr = sharpe_se_nw(pl["net"], lag)
    out["SR_gross"] = sharpe(pl["gross"])
    out["variance_ratio"] = vr
    out["turnover_mean"] = float(pl["turnover"].mean())
    gsd = pl["gross"].std()
    out["cost_drag_sd_units"] = float(pl["cost"].mean() / gsd) if gsd > 0 else float("nan")
    skew, kurt = moments(pl["net"])
    t_eff = len(pl) / vr if np.isfinite(vr) and vr > 0 else len(pl)
    out.update({"skew": skew, "kurt": kurt, "T_eff": float(t_eff)})
    out["PSR0"] = probabilistic_sharpe(sr, 0.0, t_eff, skew, kurt)
    sd_trials = sr_sd_trials if (sr_sd_trials and np.isfinite(sr_sd_trials)) else se
    out["SR_star"], out["DSR"] = deflated_sharpe(sr, t_eff, skew, kurt, n_trials, sd_trials)
    out["n_trials"] = int(n_trials)
    sh = shrink_sharpe(sr, se, float(s["prior_tau_sharpe"]))
    out["SR_shrunk"], out["SR_post_sd"], out["P_SR_pos"] = sh["post_mean"], sh["post_sd"], sh["p_positive"]

    per = {}
    for f in folds:
        per[f["period"]] = sharpe(fold_pnl([f], c0)["net"])
    vals = np.array([v for v in per.values() if np.isfinite(v)])
    out["months"] = int(len(vals))
    out["months_pos_frac"] = float((vals > 0).mean()) if len(vals) else float("nan")
    means = np.array([fold_pnl([f], c0)["net"].mean() for f in folds])
    out["months_t"] = float(means.mean() / (means.std(ddof=1) / math.sqrt(len(means)))) if len(means) > 2 and means.std(ddof=1) > 0 else float("nan")
    out["monthly_sharpe"] = per

    tl = int(s["target_lag"])
    rho = pd.concat([ic_series(f["P"], f["T"], tl) for f in folds])
    ic, ic_se = mean_se_nw(rho.to_numpy(), lag)
    out.update({"IC": ic, "IC_se": ic_se, "IC_t": ic / ic_se if ic_se and ic_se > 0 else float("nan")})
    out["IC_std"] = float(rho.std())
    out["IC_decay"] = {int(k): float(pd.concat([ic_series(f["P"], f["T"], k) for f in folds]).mean()) for k in (1, 4, 24, 96)}
    if all("disp" in f for f in folds):
        d = pd.concat([f["disp"] for f in folds]).reindex(rho.index)
        ok = d.notna() & (d > 0)
        out["IC_disp_corr"] = float(np.corrcoef(rho[ok], d[ok])[0, 1]) if ok.sum() > 10 else float("nan")
    out["concentration_raw"] = float(np.nanmean([concentration_raw(f["P"]) for f in folds]))
    fl = [flat_stats(f["P"], float(s["flat"]["rel_threshold"])) for f in folds]
    out["flat_fraction"] = float(np.mean([x["flat_fraction"] for x in fl]))
    out["longest_flat_run"] = int(max(x["longest_flat_run"] for x in fl))
    return out
