"""C5 r06 - ASSUMPTION AUDIT after novelty check #4 (Z not novel: global 58.14 deg, quality-gate 52.53 deg).

Purpose: separate MEASURED facts from MODEL assumptions before spending another server check. Measurement only on the
exact files the server judged (submissions/ + submissions/not_novel/) - no new candidate is selected here.

Per period (all 31: dev 001-029 + vault 030-031), positions on the 800 scored rows:
  probes  : the server-judged files A, AB, L240, Z, X (+ S06..S09 legs as defined in seasons/S1/occupied_directions.py)
  library : 81 generic "pot-like" constructions (feature levels/changes, extremeness, factor-residual reversal grid,
            per-period ridge models on all bases) x smoothing {0, 48, 336} - stand-ins for what other teams may have built
  switch  : SWD = dispersion-regime allocation between AB and Z (pre-declared, Nagel-2012 hypothesis: reversal pays
            more when cross-sectional dispersion is high); SWC = calendar alternation (CONTROL ONLY, never submitted)
Tests:
  T1 facts      : per-candidate SR (full / recent / vault, NW s.e.) - which numbers are in-sample or selection-touched
  T2 linearity  : is rho(X,l) ~ a*rho(AB,l) + b*rho(Z,l)?  (the assumption behind every blend prediction so far)
  T3 cross-term : among stand-ins close to Z, how correlated are they with AB (and vice versa)?
  T4 calibration: does any library statistic rank the 8 server-measured probes correctly? (if not -> do not use it)
  T5 stationarity: is the hourly correlation of AB/Z with their stand-ins the same in both dispersion regimes?
                   (the switch's novelty arithmetic assumes it)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lab import honest as H
from lab.backtest import pnl
from lab.loader import load_strategy
from lab.metrics import sharpe_se_nw
from lab.official import periods_split
from lab.paths import EXPERIMENTS, REPORTS, settings

C5 = Path(__file__).resolve().parents[1]
FILES = {
    "A": C5 / "submissions/not_novel/c5_idio_reversal_change_lag48_n_factors8_smooth_halflife168p0_smooth_window6.py",
    "AB": C5 / "submissions/not_novel/c5_idio_reversal_combo.py",
    "L240": C5 / "submissions/not_novel/c5_idio_reversal_change_lag240_n_factors3_smooth_halflife48p0_smooth_window1.py",
    "Z": C5 / "submissions/c5_orthogonal_learned.py",
    "X": C5 / "submissions/c5_orthogonal_learned_include_abTrue.py",
}
# server facts: rho at the quality gate (cos of the angle)
SERVER = {"A": 0.512, "AB": 0.638, "L240": 0.632, "Z": 0.608,
          "S06": float(np.cos(np.radians(16.31))), "S07": float(np.cos(np.radians(38.40))),
          "S08": float(np.cos(np.radians(42.34))), "S09": float(np.cos(np.radians(61.25)))}
ROWS = 800 + 6300
EMB = 127
OUT = EXPERIMENTS / "c5_r06"
OUT.mkdir(parents=True, exist_ok=True)


def eig_cov(x, W=720):
    D = np.nan_to_num(x.diff().to_numpy(float))
    T, J = D.shape
    cs1 = np.cumsum(D, axis=0); cs2 = np.cumsum(D[:, :, None] * D[:, None, :], axis=0)
    o1 = np.zeros_like(cs1); o2 = np.zeros_like(cs2)
    if T > W:
        o1[W:] = cs1[: T - W]; o2[W:] = cs2[: T - W]
    n = np.minimum(np.arange(1, T + 1), W).astype(float)
    m1 = (cs1 - o1) / n[:, None]
    cov = (cs2 - o2) / n[:, None, None] - m1[:, :, None] * m1[:, None, :]
    return np.linalg.eigh(cov)[1], n


def idio(x, V, n, K, L):
    J = x.shape[1]
    Vk = V[:, :, J - K:]
    chg = np.nan_to_num(x.diff(L).to_numpy(float))
    res = chg - np.einsum("tjk,tk->tj", Vk, np.einsum("tjk,tj->tk", Vk, chg))
    res[n < 48] = 0.0
    return pd.DataFrame(-res, index=x.index, columns=x.columns)


def ridge_model(bases, T, n_val):
    """Per-period pooled ridge on ranked bases, fit on rows before the scored window minus the embargo (honest)."""
    keys = list(bases)
    cube = np.stack([H.rank(bases[k]).fillna(0.0).to_numpy(float) for k in keys], axis=-1)  # t x j x k
    y = H.zx(T.reindex(index=bases[keys[0]].index, columns=bases[keys[0]].columns)).to_numpy(float)
    tr = slice(0, cube.shape[0] - n_val - EMB)
    Xtr, ytr = cube[tr].reshape(-1, len(keys)), y[tr].reshape(-1)
    ok = np.isfinite(ytr)
    Xtr, ytr = Xtr[ok], np.clip(ytr[ok], -5, 5)
    G = Xtr.T @ Xtr
    lam = 1e-2 * np.trace(G) / len(keys)
    w = np.linalg.solve(G + lam * np.eye(len(keys)), Xtr.T @ ytr)
    return pd.DataFrame(cube @ w, index=bases[keys[0]].index, columns=bases[keys[0]].columns)


def library(F, T, hls=(0.0, 48.0, 336.0)):
    g = H.blocks(F)
    raw = {}
    for i in range(1, 7):
        raw[f"lvl{i}"] = H.rank(g[i])
        for k in (1, 6, 24, 120, 240):
            raw[f"d{k}_{i}"] = H.rank(g[i].diff(k))
        m = g[i].rolling(168, min_periods=84)
        raw[f"tsz{i}"] = (g[i] - m.mean()) / m.std()
        raw[f"az{i}"] = -H.zx(g[i]).abs()
    V, n = eig_cov(g[1])
    for L in (6, 24, 48, 120, 240, 480):
        for K in (1, 3, 5, 8):
            raw[f"idio_F1_L{L}_K{K}"] = idio(g[1], V, n, K, L)
    for i in range(2, 7):
        Vi, ni = eig_cov(g[i])
        raw[f"idio_F{i}_L120_K3"] = idio(g[i], Vi, ni, 3, 120)
    sys.path.insert(0, str(C5 / "research"))
    bases = dict(_r05_bases(F))
    raw["ml_all"] = ridge_model(bases, T, 800)
    raw["ml_F1"] = ridge_model({k: v for k, v in bases.items() if k.endswith("_1") or k == "lvl1"}, T, 800)
    out = {}
    for k, v in raw.items():
        u = H.unit(v)
        for hl in hls:
            out[f"{k}@{int(hl)}"] = H.unit(H.smooth(u, hl)) if hl else u
    return out


def _r05_bases(F):
    g = H.blocks(F)
    z = {i: H.zx(g[i]) for i in g}
    for i in range(1, 7):
        yield f"lvl{i}", g[i]
        for k in (1, 6, 24, 120):
            yield f"d{k}_{i}", g[i].diff(k)
        yield f"az0_{i}", z[i].abs()
        yield f"az24_{i}", H.zx(g[i].diff(24)).abs()
        yield f"az120_{i}", H.zx(g[i].diff(120)).abs()
        yield f"zxd24_{i}", z[i] * H.zx(g[i].diff(24))
        d1 = g[i].diff(1)
        yield f"vol24_{i}", d1.rolling(24, min_periods=12).std()
        yield f"vol168_{i}", d1.rolling(168, min_periods=84).std()
        m = g[i].rolling(168, min_periods=84)
        yield f"tsz_{i}", (g[i] - m.mean()) / m.std()
    for i in (1, 2, 3, 4):
        yield f"sp{i}_6", z[i] - z[6]
        yield f"sp{i}_5", z[i] - z[5]
    yield "sp5_6", z[5] - z[6]


def ewma(s: pd.Series, hl: float) -> pd.Series:
    return s.ewm(halflife=hl, adjust=False).mean()


def regime_weight(F):
    """SWD weight on AB: 1 when cross-sectional dispersion of 24h Feature.1 changes is above its trailing 720h median."""
    d = H.blocks(F)[1].diff(24).std(axis=1)
    med = d.rolling(720, min_periods=240).median()
    hi = (d > med).astype(float).where(med.notna(), 0.5)
    return ewma(hi, 24.0), hi


def calendar_weight(F):
    idx = pd.Series(np.asarray(F.index, dtype=np.int64), index=F.index)
    return ewma(((idx // 168) % 2 == 0).astype(float), 24.0)


def blend(wt, P1, P2):
    w = wt.to_numpy()[:, None]
    return H.unit(pd.DataFrame(w * P1.to_numpy() + (1 - w) * P2.to_numpy(), index=P1.index, columns=P1.columns))


def rowcos(P, Q):
    a, b = P.to_numpy(float), Q.reindex_like(P).to_numpy(float)
    num = (a * b).sum(1)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def main():
    devp, vp = periods_split(settings())
    periods = devp + vp
    out = OUT
    hls = (0.0, 48.0, 336.0)
    if "--smoke" in sys.argv:
        periods = periods[-1:]
    if "--slow" in sys.argv:  # added after T3: no stand-in reached |rho| 0.4 with Z (hl 1344) -> slow-smoothed stand-ins
        out, hls = OUT / "slow", (672.0, 1344.0)
        out.mkdir(parents=True, exist_ok=True)
    cls = {k: load_strategy(v) for k, v in FILES.items()}
    hl_leg = H.leg_hl()
    per_rows, cosrec, regrec = [], {}, []
    for p in periods:
        D = H._load(p, ROWS)
        F, Xr, T = D["features"], D["returns"], D["target"]
        vi = F.index[len(F) - 800:]
        P = {}
        for k, C in cls.items():
            m = C()
            S, _ = m._score(F)
            P[k] = H.unit(m._positions(S).reindex(columns=Xr.columns).fillna(0.0)).loc[vi]
        raw = H.occupied_raw(F)
        for k, v in raw.items():
            P[k] = H.unit(H.smooth(v.reindex(columns=Xr.columns), hl_leg[k])).loc[vi]
        wD, hi = regime_weight(F)
        P["SWD"] = blend(wD.loc[vi], P["AB"], P["Z"])
        P["SWC"] = blend(calendar_weight(F).loc[vi], P["AB"], P["Z"])
        lib = {k: v.reindex(columns=Xr.columns).fillna(0.0).loc[vi] for k, v in library(F, T, hls).items()}
        for k, Pk in P.items():
            r = pnl(Pk, Xr.loc[vi], 5.0)["net"].reset_index(drop=True)
            per_rows.append({"period": p, "cand": k, "mean": r.mean(), "sd": r.std(), "n": len(r)})
            np.save(out / f"pnl_{k}_{p}.npy", r.to_numpy())
        cands = list(P)
        allsig = {**P, **lib}
        for c in cands:
            for l, Q in allsig.items():
                cosrec.setdefault((c, l), []).append(rowcos(P[c], Q))
        regrec.append(pd.DataFrame({"period": p, "w": wD.loc[vi].to_numpy(), "hi": hi.loc[vi].to_numpy()}))
        print(p, "done", flush=True)
    keys = sorted(cosrec)
    M = {f"{c}|{l}": np.concatenate(v).astype(np.float32) for (c, l), v in cosrec.items()}
    np.savez_compressed(out / "hourly_cos.npz", **M)
    pd.concat(regrec, ignore_index=True).to_parquet(out / "regime.parquet")
    pd.DataFrame(per_rows).to_csv(out / "period_pnl.csv", index=False)
    print("saved", len(keys), "pairs")


if __name__ == "__main__":
    main()
