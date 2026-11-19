"""Synthetic stand-in panel with the SAME file layout as the competition package.

Purpose: test the research/validation tooling end-to-end before (and independently of) the
real data. It is NOT competition data, carries no information about it, and must never be
used to fit or tune a submission (the Rules forbid external data; this is used only to
check that the machinery recovers a planted edge and rejects noise).

Planted structure (known ground truth):
  * alpha(t) = demean(tanh(z F1 * z F2)) + 0.7 * regime(t) * demean(z(F6 - mean24 F6))
    -> nonlinear interaction + regime-gated change; IC ~ few %.
  * an "obvious" reversal effect in X that the target is NEUTRALISED against,
    so the textbook reversal signal pays on X but has ~0 IC on the target.
  * target = clip(xs z( residual of sum_{k=0}^{h-1} X(t+k) on reversal ), +-5), h = 8.
  * fat tails (Student-t 5), volatility regimes, heterogeneous feature scales,
    warm-up zeros, cumulative period files with per-period ticker rotation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

H_TRUE = 8


def _ar(rng, T, J, phi, sd=1.0):
    out = np.empty((T, J))
    x = rng.normal(0, sd / np.sqrt(1 - phi * phi), J)
    eps = rng.normal(0, sd, (T, J))
    for t in range(T):
        x = phi * x + eps[t]
        out[t] = x
    return out


def _z(a):
    d = a - np.nanmean(a, axis=1, keepdims=True)
    s = np.nanstd(d, axis=1, keepdims=True)
    return np.divide(d, s, out=np.zeros_like(d), where=s > 0)


def _trail_mean(a, w):
    c = np.cumsum(np.vstack([np.zeros((1, a.shape[1])), a]), axis=0)
    out = np.empty_like(a)
    for t in range(a.shape[0]):
        lo = max(0, t + 1 - w)
        out[t] = (c[t + 1] - c[lo]) / (t + 1 - lo)
    return out


def generate(months: int = 14, J: int = 20, seed: int = 7, start: str = "2025-06-01"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=months * 30 * 24, freq="h")
    T = len(idx)

    a1 = _ar(rng, T, J, 0.995, 0.10)
    a2 = _ar(rng, T, J, 0.97, 0.25)
    f6 = _ar(rng, T, J, 0.999, 0.05) + _ar(rng, T, 1, 0.999, 0.05)
    regime_state = _ar(rng, T, 1, 0.995, 0.10)[:, 0]
    vol = 0.012 * np.exp(0.6 * np.tanh(regime_state))  # calibrated: raw hourly rebalancing costs ~0.06 SR (as the rules state)
    regime = (regime_state > 0).astype(float)

    F1 = a1 + 0.05 * rng.normal(size=(T, J))
    F2 = a2
    F6 = f6
    chg6 = _z(F6 - _trail_mean(F6, 24))
    alpha = np.tanh(_z(F1) * _z(F2))
    alpha = alpha - alpha.mean(axis=1, keepdims=True)
    alpha = alpha + 0.7 * regime[:, None] * chg6
    alpha = _z(alpha)

    beta = rng.uniform(0.6, 1.4, J)
    mkt = rng.normal(0, 1, T)
    eps = rng.standard_t(5, (T, J)) / np.sqrt(5 / 3)
    X = np.zeros((T, J))
    for t in range(1, T):
        rev = -0.03 * _z(X[max(0, t - 6):t].sum(axis=0, keepdims=True))[0]
        X[t] = vol[t] * (0.6 * beta * mkt[t] + 0.045 * alpha[t - 1] + rev + eps[t])

    csx = np.vstack([np.zeros((1, J)), np.cumsum(X, axis=0)])
    F3 = csx[1:] - csx[np.maximum(np.arange(1, T + 1) - 24, 0)]  # trailing 24h return sum
    scale = 10 ** rng.uniform(0, 2, J)
    F4 = np.exp(_ar(rng, T, J, 0.95, 0.2)) * scale
    F5 = F1 + 0.5 * rng.normal(size=(T, J)) * 0.1

    # target: forward window sum (offset 0) residualised on trailing-6h reversal, z, clip
    cs = np.vstack([np.zeros((1, J)), np.cumsum(X, axis=0)])
    R = np.full((T, J), np.nan)
    R[: T - H_TRUE + 1] = cs[H_TRUE:] - cs[: T - H_TRUE + 1]
    rev6 = np.full((T, J), 0.0)
    rev6[6:] = cs[6:T] - cs[: T - 6]
    rz = _z(rev6)
    Rd = R - np.nanmean(R, axis=1, keepdims=True)
    beta_r = np.nansum(Rd * rz, axis=1, keepdims=True) / np.maximum((rz * rz).sum(axis=1, keepdims=True), 1e-12)
    Rres = Rd - beta_r * rz
    disp = np.nanstd(Rres, axis=1)
    tgt = np.clip(np.divide(Rres, disp[:, None], out=np.zeros_like(Rres), where=disp[:, None] > 0), -5, 5)
    tgt = np.nan_to_num(tgt)
    tgt[:48] = 0.0
    disp[:48] = np.nan

    feats = {"Feature.1": F1, "Feature.2": F2, "Feature.3": F3, "Feature.4": F4, "Feature.5": F5, "Feature.6": F6}
    return idx, feats, X, tgt, disp


def write_package(out_root: Path, months: int = 14, n_periods: int = 6, J: int = 20, seed: int = 7) -> list[str]:
    idx, feats, X, tgt, disp = generate(months, J, seed)
    out_root.mkdir(parents=True, exist_ok=True)
    month_ends = pd.Series(idx).groupby(idx.to_period("M")).max().tolist()
    rng = np.random.default_rng(seed + 1)
    written = []
    for k, end in enumerate(month_ends[len(month_ends) - n_periods:], start=1):
        n = int((idx <= end).sum())
        perm = rng.permutation(J)
        names = [f"ticker.{perm[j] + 1}" for j in range(J)]
        cols = pd.MultiIndex.from_product([list(feats), names], names=["feature", "ticker"])
        F = pd.DataFrame(np.hstack([feats[f][:n] for f in feats]), index=idx[:n], columns=cols)
        nnn = f"{k:03d}"
        F.to_parquet(out_root / f"{nnn}.small.features.parquet")
        pd.DataFrame(X[:n], index=idx[:n], columns=names).to_parquet(out_root / f"{nnn}.small.returns.parquet")
        pd.DataFrame(tgt[:n], index=idx[:n], columns=names).to_parquet(out_root / f"{nnn}.small.target.parquet")
        pd.DataFrame({"dispersion": disp[:n]}, index=idx[:n]).to_parquet(out_root / f"{nnn}.small.dispersion.parquet")
        written.append(nnn)
    return written
