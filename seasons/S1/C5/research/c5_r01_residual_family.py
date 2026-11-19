"""C5 r01 - factor-residual reversal family: PRE-DECLARED grid of 16 (feature F1, W 720h).

  L (change horizon) in {24, 48, 120, 240}  x  K (removed factors) in {3, 8}  x  hl (smoothing) in {48, 168}
Ranked by SR5_proxy (10 most recent periods: dev 022-029 + vault 030-031); every call is a logged trial.
Novelty vs our own pot entry = cos09 (S09 as submitted: L120, K8, hl672) must be <= 0.5 (target <= 0.45).
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lab.honest import admissible_v2, blocks, evaluate, unit
from lab.paths import REPORTS


def idio_rev(F, W=720, K=8, L=120):
    x = blocks(F)[1]
    D = np.nan_to_num(x.diff().to_numpy(float))
    T, J = D.shape
    cs1 = np.cumsum(D, axis=0); cs2 = np.cumsum(D[:, :, None] * D[:, None, :], axis=0)
    o1 = np.zeros_like(cs1); o2 = np.zeros_like(cs2)
    if T > W:
        o1[W:] = cs1[: T - W]; o2[W:] = cs2[: T - W]
    n = np.minimum(np.arange(1, T + 1), W).astype(float)
    m1 = (cs1 - o1) / n[:, None]
    cov = (cs2 - o2) / n[:, None, None] - m1[:, :, None] * m1[:, None, :]
    V = np.linalg.eigh(cov)[1][:, :, J - K:]
    chg = np.nan_to_num(x.diff(L).to_numpy(float))
    res = chg - np.einsum("tjk,tk->tj", V, np.einsum("tjk,tj->tk", V, chg))
    res[n < 48] = 0.0
    return pd.DataFrame(-res, index=x.index, columns=x.columns)


rows = []
for L, K, hl in itertools.product((24, 48, 120, 240), (3, 8), (48.0, 168.0)):
    name = f"c5r01_L{L}_K{K}_hl{int(hl)}"
    m = evaluate(lambda F, L=L, K=K: idio_rev(F, 720, K, L), name, hl=hl, notes="C5 r01 residual family grid", verbose=False)
    m["v2"] = admissible_v2(m)
    rows.append(m)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
                      if k in ("name", "SR5", "SR10", "SR5_recent", "vault_SR5", "SR5_proxy", "se_proxy", "proxy_periods_pos",
                               "cos06", "cos07", "cos08", "cos09", "gate_inSR", "DSR", "v2")}), flush=True)
df = pd.DataFrame(rows).sort_values("SR5_proxy", ascending=False)
REPORTS.mkdir(parents=True, exist_ok=True)
df.to_csv(REPORTS / "c5_r01_residual_family.csv", index=False)
print(df[["name", "SR5", "SR10", "SR5_proxy", "se_proxy", "proxy_periods_pos", "vault_SR5", "cos09", "cos08", "cos06", "gate_inSR", "DSR"]]
      .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
