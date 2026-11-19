"""C5 r05 - genuine orthogonal component Z + combination X (PRE-DECLARED: exactly 2 evaluations).

Z = C4 multi-agent offline-learned model (grouped PCA-5 weights, hl 1344), residualised per hour on the occupied legs.
    Weights were fit ONLY on dev periods 001-029 -> vault 030-031 is a clean out-of-sample test for Z.
X = unit( unit(AB positions) + unit(Z positions) )  (equal risk). If rho(Z,AB) ~ 0 then rho(X,AB) ~ 0.71 -> occupant
    correlation ~ 0.64 * 0.71 ~ 0.45 (AB's measured occupant rho is 0.638).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lab import honest as H
from lab.honest import admissible_v2, evaluate
from lab.loader import load_strategy

W = json.load(open(Path("../C4/research/agents/_offline_learned/weights_g_pca5_1344.json")))["w_base"]
W = {k: float(v) for k, v in W.items() if abs(v) > 0}


def base_panels(F):
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


def z_positions(F):
    cols = list(dict.fromkeys(F.columns.get_level_values(-1)))
    acc = 0.0
    for k, v in base_panels(F):
        if k in W:
            acc = acc + W[k] * H.unit(H.rank(v.reindex(columns=cols)))
    raw = H.occupied_raw(F)
    legs = [H.smooth(raw[k].reindex(columns=cols), H.leg_hl()[k]) for k in ("S06", "S07", "S08")]
    return H.unit(H.residualize(H.smooth(acc, 1344.0), legs))


AB = load_strategy("strategies/c5_idio_reversal_combo.py")


def ab_positions(F):
    m = AB()
    S, _ = m._score(F)
    return H.unit(m._positions(S))


def x_positions(F):
    return H.unit(ab_positions(F) + z_positions(F))


keys = ("SR5", "se", "ci_lo", "SR10", "periods_pos", "SR5_recent", "vault_SR5", "vault_IC_raw", "SR5_proxy", "se_proxy",
        "proxy_periods_pos", "IC_T", "IC_t", "cos06", "cos07", "cos08", "cos09", "gate_inSR", "DSR")
for name, fn in (("C5_Z_offline_pca5_1344", z_positions), ("C5_X_AB_plus_Z", x_positions)):
    m = evaluate(fn, name, hl=0.0, warm=4 * 1344 + 900, notes="C5 r05 pre-declared", verbose=False)
    print(name, json.dumps({k: (round(m[k], 4) if isinstance(m.get(k), float) else m.get(k)) for k in keys}), "v2", admissible_v2(m), flush=True)

# overlaps on the 10 recent periods (positions as scored)
from lab.metrics import row_cos
from lab.official import periods_split
from lab.paths import settings
devp, vp = periods_split(settings())
Pz, Pab, Px = [], [], []
for p in devp[-8:] + vp:
    D = H._load(p, 800 + 4 * 1344 + 900)
    F = D["features"]
    vi = F.index[len(F) - 800:]
    for lst, f in ((Pz, z_positions), (Pab, ab_positions), (Px, x_positions)):
        v = f(F).loc[vi].copy(); v.index = [f"{p}:{i}" for i in v.index]; lst.append(v)
Z, A_B, X = pd.concat(Pz), pd.concat(Pab), pd.concat(Px)
print(f"rho(Z,AB) {row_cos(Z, A_B).mean():+.3f} | rho(X,AB) {row_cos(X, A_B).mean():+.3f} -> predicted occupant rho {0.638 * row_cos(X, A_B).mean():.3f}")
