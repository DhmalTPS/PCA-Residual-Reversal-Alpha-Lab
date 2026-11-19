"""Season S1 - directions known to be OCCUPIED (measured by server novelty checks), used by common/lab/honest.py.

Each season gets its own file seasons/<S>/occupied_directions.py with the same three names:
  occupied_raw(F) -> {name: unsmoothed unit score (rows x tickers)}   exact definitions of our probe signals
  LEG_HL          -> {name: smoothing half-life used when comparing}
  SERVER_ANGLES   -> {name: server-measured angle (deg) between that probe and its nearest occupant}
A new season starts with an empty set (nothing known yet) and grows as novelty checks come back.

S1 evidence (S1/C4 server novelty checks, see seasons/S1/C4/config/leaderboard.json):
  S06 level reversal composite      quality-gate 16.31 deg (legacy pot)
  S07 extremeness (v1 25.12, v2 38.40 deg; v2 definition below)
  S08 F5 + 5-day change reversal    42.34 deg
  S09 PCA-idiosyncratic reversal    ACCEPTED (61.25 deg) -> itself occupies space for the rest of S1 (our own pot entry)
"""
from __future__ import annotations

import numpy as np

from lab.honest import blocks, rank, unit, zx

LEG_HL = {"S06": 336.0, "S07": 672.0, "S08": 672.0, "S09": 672.0}
SERVER_ANGLES = {"S06": 16.31, "S07": 25.12, "S08": 42.34, "S09": 0.0}


def _pca_idio_reversal(F, W=720, K=8, L=120):
    g = blocks(F)
    x = g[1]
    D = np.nan_to_num(x.diff().to_numpy(float))
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
    V = np.linalg.eigh(cov)[1][:, :, J - K:]
    chg = np.nan_to_num(x.diff(L).to_numpy(float))
    res = chg - np.einsum("tjk,tk->tj", V, np.einsum("tjk,tj->tk", V, chg))
    res[n < 48] = 0.0
    import pandas as pd
    return pd.DataFrame(-res, index=x.index, columns=x.columns)


def occupied_raw(F):
    g = blocks(F)
    r = {i: rank(g[i]) for i in g}
    s06 = unit(-(r[1] + r[2] + r[3] + r[4]) / 4 + 0.5 * r[6] - 0.5 * r[5] - 0.5 * rank(g[1].diff(120)))
    s07 = unit(-sum(rank(zx(g[i]).abs()) + rank(zx(g[i].diff(24)).abs()) + rank(zx(g[i].diff(120)).abs()) for i in (1, 2, 3, 4)))
    s08 = unit(unit(-r[5]) + unit(-sum(rank(g[i].diff(120)) for i in (1, 2, 3, 4))))
    s09 = unit(_pca_idio_reversal(F))
    return {"S06": s06, "S07": s07, "S08": s08, "S09": s09}
