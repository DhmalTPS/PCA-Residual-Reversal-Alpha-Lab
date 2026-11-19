"""PEER_FACTOR family - cross-asset structure (trailing correlation network / trailing PCA / betas).

PLAN (written before any returns-based evaluation; HARD LIMIT 12 lab.honest.evaluate calls):
  Stage 0 (free, features-only, NO returns/targets touched): geometry pre-screen.  For ~15 constructions compute
          the hl-672 smoothed score on the last 800 rows of 6 dev files and its mean per-hour cosine with the
          cached occupied legs S06/S07/S08.  Constructions with max|cos| > 0.35 are near an occupant by construction
          and are only evaluated if they carry a distinct hypothesis.  This screen uses no performance information.
  Stage 1 (<= 8 evaluate calls): one call per pre-declared hypothesis with its PRIOR sign:
          E1 pca_idio_rev   : -(120h change of F1 residual to top-3 trailing-720h PCs of hourly dF1)
          E2 bab            : -beta to the equal-weight dF1 factor (trailing 720h)
          E3 netmom         : +peer-weighted 120h change (positive-corr network, 720h), own change projected out
          E4 idio_vol       : -trailing 720h std of PCA-residual dF1
          E5 factor_timing  : +beta_i x factor 120h move (PC1..3 loadings x factor returns)
          E6 centrality     : -mean trailing correlation to peers (isolated assets)
          (+ up to 2 more chosen from stage-0 geometry, declared in the log before running)
  Stage 2 (<= 4 calls): refinements ONLY of a stage-1 hypothesis with SR5 > 0 and ci_lo > -0.005
          (window / hl), each counted.  Sign flips count as trials.
Every evaluate() is logged to the registry (DSR uses the honest count).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lab import honest as h  # noqa: E402
from lab.metrics import row_cos  # noqa: E402


# ----------------------------------------------------------------------------- causal cross-asset primitives
def _roll_moments(D: np.ndarray, W: int, minp: int = 48):
    """Trailing-window (exactly the last W rows, inclusive) mean and covariance of D (T, J)."""
    T, J = D.shape
    cs1 = np.cumsum(D, axis=0)
    cs2 = np.cumsum(D[:, :, None] * D[:, None, :], axis=0)
    o1 = np.zeros_like(cs1)
    o2 = np.zeros_like(cs2)
    if T > W:
        o1[W:] = cs1[: T - W]
        o2[W:] = cs2[: T - W]
    n = np.minimum(np.arange(1, T + 1), W).astype(float)
    m1 = (cs1 - o1) / n[:, None]
    cov = (cs2 - o2) / n[:, None, None] - m1[:, :, None] * m1[:, None, :]
    ok = n >= minp
    return m1, cov, ok


def _corr(cov):
    sd = np.sqrt(np.clip(np.einsum("tjj->tj", cov), 0.0, None))
    den = sd[:, :, None] * sd[:, None, :]
    return np.divide(cov, den, out=np.zeros_like(cov), where=den > 1e-18)


def _base_change(F: pd.DataFrame, kind: str = "f1") -> tuple[np.ndarray, pd.Index, list]:
    g = h.blocks(F)
    tick = list(g[1].columns)
    if kind == "f1":
        D = g[1].diff().to_numpy(float)
    else:  # mean of scaled changes of F1-4
        parts = []
        for i in (1, 2, 3, 4):
            d = g[i].diff()
            s = d.rolling(168, min_periods=24).std()
            parts.append((d / s.where(s > 0)).to_numpy(float))
        D = np.nanmean(np.stack(parts), axis=0)
    return np.nan_to_num(D), g[1].index, tick


def _pca_basis(cov, K):
    """Top-K eigenvectors per hour (T, J, K)."""
    w, V = np.linalg.eigh(cov)  # ascending
    return V[:, :, ::-1][:, :, :K]


def _project_out(x, V):
    """x (T,J) minus projection on columns of V (T,J,K) (orthonormal)."""
    c = np.einsum("tjk,tj->tk", V, x)
    return x - np.einsum("tjk,tk->tj", V, c)


def _df(a, idx, tick):
    return pd.DataFrame(a, index=idx, columns=tick)


# ----------------------------------------------------------------------------- constructions
def pca_idio_rev(F, W=720, K=3, L=120, kind="f1"):
    D, idx, tick = _base_change(F, kind)
    _, cov, ok = _roll_moments(D, W)
    V = _pca_basis(cov, K)
    g = h.blocks(F)
    chg = np.nan_to_num(g[1].diff(L).to_numpy(float)) if kind == "f1" else \
        np.nan_to_num(pd.DataFrame(D).rolling(L, min_periods=L // 2).sum().to_numpy())
    res = _project_out(chg, V)
    res[~ok] = 0.0
    return -_df(res, idx, tick)


def bab(F, W=720, kind="f1"):
    D, idx, tick = _base_change(F, kind)
    m = D.mean(axis=1, keepdims=True)
    A = np.concatenate([D, m], axis=1)
    _, cov, ok = _roll_moments(A, W)
    J = D.shape[1]
    beta = cov[:, :J, J] / np.where(cov[:, J, J] > 0, cov[:, J, J], np.nan)[:, None]
    beta[~ok] = np.nan
    return -_df(beta, idx, tick)


def peer_weights(D, W=720):
    _, cov, ok = _roll_moments(D, W)
    C = _corr(cov)
    Wt = np.clip(C, 0.0, None)
    J = D.shape[1]
    Wt[:, np.arange(J), np.arange(J)] = 0.0
    rs = Wt.sum(axis=2, keepdims=True)
    uni = (1.0 - np.eye(J)) / (J - 1)
    Wt = np.where(rs > 1e-12, Wt / np.where(rs > 1e-12, rs, 1.0), uni[None])
    return Wt, C, ok


def netmom(F, W=720, L=120, kind="f1", resid_own=True):
    D, idx, tick = _base_change(F, kind)
    Wt, _, ok = peer_weights(D, W)
    g = h.blocks(F)
    own = h.zx(g[1].diff(L)).fillna(0.0).to_numpy(float)
    peer = np.einsum("tjk,tk->tj", Wt, own)
    out = _df(peer, idx, tick)
    if resid_own:
        out = h.residualize(out, [_df(own, idx, tick)])
    out[~ok] = 0.0
    return out


def idio_vol(F, W=720, K=3, kind="f1"):
    D, idx, tick = _base_change(F, kind)
    _, cov, ok = _roll_moments(D, W)
    V = _pca_basis(cov, K)
    R = _project_out(D - D.mean(axis=1, keepdims=True), V)
    iv = pd.DataFrame(R).rolling(W, min_periods=48).std().to_numpy()
    iv[~ok] = np.nan
    return -_df(iv, idx, tick)


def factor_timing(F, W=720, K=3, L=120, kind="f1"):
    """beta_i(k) x factor k's trailing L-hour move (sum over k): the factor-explained part of momentum."""
    D, idx, tick = _base_change(F, kind)
    _, cov, ok = _roll_moments(D, W)
    V = _pca_basis(cov, K)
    fr = np.einsum("tjk,tj->tk", V, D)  # factor returns with time-t loadings (causal)
    frL = pd.DataFrame(fr).rolling(L, min_periods=L // 2).sum().to_numpy()
    s = np.einsum("tjk,tk->tj", V, np.nan_to_num(frL))
    s[~ok] = 0.0
    return _df(s, idx, tick)


def centrality(F, W=720, kind="f1"):
    D, idx, tick = _base_change(F, kind)
    _, cov, ok = _roll_moments(D, W)
    C = _corr(cov)
    J = D.shape[1]
    c = (C.sum(axis=2) - 1.0) / (J - 1)
    c[~ok] = np.nan
    return -_df(c, idx, tick)


def leadlag(F, W=720, kind="f1", L=24):
    """Lagged-peer predictability: score_i = sum_j b_ij * dx_j over last L h, with b_ij = trailing corr(dx_i(t), dx_j(t-1))."""
    D, idx, tick = _base_change(F, kind)
    Dl = np.vstack([np.zeros((1, D.shape[1])), D[:-1]])
    J = D.shape[1]
    A = np.concatenate([D, Dl], axis=1)
    _, cov, ok = _roll_moments(A, W)
    C = _corr(cov)[:, :J, J:]  # corr(own_t, peer_{t-1})
    C[:, np.arange(J), np.arange(J)] = 0.0
    rec = pd.DataFrame(D).rolling(L, min_periods=L // 2).sum().to_numpy()
    s = np.einsum("tjk,tk->tj", C, np.nan_to_num(rec))
    s[~ok] = 0.0
    return _df(s, idx, tick)


def peer_gap_level(F, W=720, kind="f1"):
    """F1 level gap to the peer neighbourhood (asset above its peers -> reverts)."""
    D, idx, tick = _base_change(F, kind)
    Wt, _, ok = peer_weights(D, W)
    g = h.blocks(F)
    lv = h.zx(g[1]).fillna(0.0).to_numpy(float)
    gap = lv - np.einsum("tjk,tk->tj", Wt, lv)
    gap[~ok] = 0.0
    return -_df(gap, idx, tick)


CANDS = {
    "pca_idio_rev": lambda F: pca_idio_rev(F),
    "pca_idio_rev_K5": lambda F: pca_idio_rev(F, K=5),
    "pca_idio_rev_m4": lambda F: pca_idio_rev(F, kind="m4"),
    "bab": lambda F: bab(F),
    "bab_m4": lambda F: bab(F, kind="m4"),
    "netmom": lambda F: netmom(F),
    "netmom_raw": lambda F: netmom(F, resid_own=False),
    "idio_vol": lambda F: idio_vol(F),
    "factor_timing": lambda F: factor_timing(F),
    "factor_timing_L24": lambda F: factor_timing(F, L=24),
    "centrality": lambda F: centrality(F),
    "leadlag": lambda F: leadlag(F),
    "peer_gap_level": lambda F: peer_gap_level(F),
}


def geometry(names, periods=("004", "009", "014", "019", "024", "029"), hl=672.0):
    warm = int(4 * hl + 1200)
    rows = []
    for p in periods:
        D = h._load(p, h.VAL + warm)
        F = D["features"]
        vi = F.index[len(F) - h.VAL:]
        legs = h.occupied_smoothed(p, int(4 * 672 + 600))
        sm = {}
        for n in names:
            U = h.unit(CANDS[n](F).reindex(columns=legs["S06"].columns))
            sm[n] = h.smooth(U, hl).loc[vi]
            r = {"p": p, "name": n}
            for k in ("S06", "S07", "S08"):
                r[k] = float(row_cos(sm[n], legs[k]).mean())
            rows.append(r)
        del D, F
    T = pd.DataFrame(rows).groupby("name")[["S06", "S07", "S08"]].mean()
    T["max"] = T.abs().max(axis=1)
    return T.sort_values("max")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "geom"
    if mode == "geom":
        pd.set_option("display.width", 200)
        print(geometry(list(CANDS)).round(3))
    elif mode == "eval":
        import json
        res = []
        for n in sys.argv[2:]:
            m = h.evaluate(CANDS[n], f"peer_{n}", hl=672.0, warm=int(4 * 672 + 1200), notes="PEER_FACTOR agent")
            m["admissible"] = bool(h.admissible(m))
            res.append(m)
        out = ROOT / "research" / "agents" / "peer_factor_results.jsonl"
        with open(out, "a") as f:
            for m in res:
                f.write(json.dumps({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}) + "\n")
