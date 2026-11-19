"""Self-contained final constructions of the PEER_FACTOR agent (PCA idiosyncratic 120h reversal)."""
import numpy as np
import pandas as pd
from lab.honest import blocks, occupied_raw, residualize, unit


def pca_idio_rev(F, W=720, K=8, L=120):
    """-(120h change of Feature.1 minus its projection on the top-K eigenvectors of the trailing-W-hour
    covariance of hourly Feature.1 changes).  Causal: row t uses rows t-W+1..t only."""
    g = blocks(F)
    x = g[1]
    D = np.nan_to_num(x.diff().to_numpy(float))
    T, J = D.shape
    cs1 = np.cumsum(D, axis=0)
    cs2 = np.cumsum(D[:, :, None] * D[:, None, :], axis=0)
    o1 = np.zeros_like(cs1); o2 = np.zeros_like(cs2)
    if T > W:
        o1[W:] = cs1[:T - W]; o2[W:] = cs2[:T - W]
    n = np.minimum(np.arange(1, T + 1), W).astype(float)
    m1 = (cs1 - o1) / n[:, None]
    cov = (cs2 - o2) / n[:, None, None] - m1[:, :, None] * m1[:, None, :]
    V = np.linalg.eigh(cov)[1][:, :, ::-1][:, :, :K]
    chg = np.nan_to_num(x.diff(L).to_numpy(float))
    res = chg - np.einsum("tjk,tk->tj", V, np.einsum("tjk,tj->tk", V, chg))
    res[n < 48] = 0.0
    return -pd.DataFrame(res, index=x.index, columns=x.columns)


def fn_K8_W720(F):
    return pca_idio_rev(F, W=720, K=8)


def fn_K8_W720_perp8(F):
    return residualize(unit(pca_idio_rev(F, W=720, K=8)), [occupied_raw(F)["S08"]])


def fn_K5_W720(F):
    return pca_idio_rev(F, W=720, K=5)
