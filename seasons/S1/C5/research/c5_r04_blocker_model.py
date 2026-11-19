"""C5 r04 - blocker triangulation + constrained selection (PRE-REGISTERED rule, 2026-10-15 after 2 novelty readings).

Readings: cos(o,A)=0.512 (59.18 deg), cos(o,AB)=0.638 (50.38 deg); AB=(A+B)/|.|, rho(A,B)=0.41 =>
  blocker projection on span{A,B} = 0.339*A + 0.421*B  ->  predicted cos(o,X) = 0.339*rho(X,A) + 0.421*rho(X,B).
rho measured over ALL 31 validation blocks (the server's rho is over the full scored history).
Candidates: idio-reversal configs already evaluated (r01 grid + r03 neighbours) + tilts unit(A_s + beta*B_s), beta in
{-0.1,-0.2,-0.3}. Rule: max recent proxy s.t. pred_cos <= 0.40, rho(S09 submitted) <= 0.45, period-bootstrap p05 > 0.
"""
import numpy as np
import pandas as pd

from lab import registry
from lab.backtest import pnl, sharpe
from lab.honest import _load, smooth, unit
from lab.metrics import row_cos, sharpe_se_nw
from lab.official import periods_split
from lab.paths import REPORTS, settings

ns = {}
exec(open("research/c5_r01_residual_family.py", encoding="utf-8").read().split("rows = []")[0], ns)
idio_rev = ns["idio_rev"]
GRID = [(L, K, hl) for L in (24, 48, 120, 240) for K in (3, 8) for hl in (48.0, 168.0)]
NB = [(36, 8, 168.0), (72, 8, 168.0), (48, 6, 168.0), (48, 10, 168.0), (48, 8, 120.0), (48, 8, 240.0),
      (96, 3, 48.0), (144, 3, 48.0), (120, 2, 48.0), (120, 4, 48.0), (120, 3, 36.0), (120, 3, 72.0)]
CFG = {f"L{L}_K{K}_hl{int(hl)}": ("single", L, K, hl) for L, K, hl in GRID + NB}
for b in (-0.1, -0.2, -0.3):
    CFG[f"A{b:+.1f}B"] = ("tilt", b)
S09 = (120, 8, 672.0)
devp, vp = periods_split(settings())
allp = devp + vp
recent = devp[-8:] + vp
pos = {k: [] for k in CFG}
pA, pB, p9 = [], [], []
r5 = {k: {} for k in CFG}
r10 = {k: {} for k in CFG}
for p in allp:
    D = _load(p, 800 + 4 * 672 + 900)
    F, X = D["features"], D["returns"]
    vi = F.index[len(F) - 800:]
    raw = {}
    def leg(L, K, hl):
        if (L, K) not in raw:
            raw[(L, K)] = unit(idio_rev(F, 720, K, L))
        return unit(smooth(raw[(L, K)], hl)).loc[vi]
    A, B, S = leg(48, 8, 168.0), leg(120, 3, 48.0), leg(*S09)
    key = [f"{p}:{i}" for i in vi]
    for lst, v in ((pA, A), (pB, B), (p9, S)):
        v = v.copy(); v.index = key; lst.append(v)
    for k, c in CFG.items():
        P = leg(c[1], c[2], c[3]) if c[0] == "single" else unit(A + c[1] * B)
        P = P.sub(P.mean(axis=1), axis=0)
        r5[k][p] = pnl(P, X.loc[vi], 5.0)["net"].reset_index(drop=True)
        r10[k][p] = pnl(P, X.loc[vi], 10.0)["net"].reset_index(drop=True)
        q = P.copy(); q.index = key; pos[k].append(q)
    print(f"[{p}]", end="", flush=True)
print()
A, B, S = pd.concat(pA), pd.concat(pB), pd.concat(p9)
rng = np.random.default_rng(20261015)
rows = []
for k in CFG:
    X = pd.concat(pos[k])
    ra, rb, r9 = row_cos(X, A).mean(), row_cos(X, B).mean(), row_cos(X, S).mean()
    d5 = r5[k]
    bs = [sharpe(pd.concat([d5[p] for p in rng.choice(recent, len(recent))], ignore_index=True)) for _ in range(1000)]
    rows.append({"config": k, "proxy5": sharpe_se_nw(pd.concat([d5[p] for p in recent], ignore_index=True), 48)[0],
                 "proxy10": sharpe_se_nw(pd.concat([r10[k][p] for p in recent], ignore_index=True), 48)[0],
                 "full5": sharpe_se_nw(pd.concat([d5[p] for p in devp], ignore_index=True), 48)[0],
                 "boot_p05": float(np.quantile(bs, 0.05)), "rhoA": ra, "rhoB": rb, "rhoS09": r9,
                 "pred_cos_blocker": 0.339 * ra + 0.421 * rb})
df = pd.DataFrame(rows)
df["pred_deg"] = np.degrees(np.arccos(df["pred_cos_blocker"].clip(-1, 1)))
df["eligible"] = (df["pred_cos_blocker"] <= 0.40) & (df["rhoS09"] <= 0.45) & (df["boot_p05"] > 0)
df = df.sort_values("proxy5", ascending=False)
df.to_csv(REPORTS / "c5_r04_blocker_model.csv", index=False)
for k in CFG:
    if CFG[k][0] == "tilt":
        registry.append({"run_id": f"c5r04_{k}", "strategy": "c5_r04_blocker_model", "mode": "screen", "data": "real-official",
                         "SR_net_5bp": float(df.loc[df.config == k, "full5"].iloc[0]), "notes": "tilt A+beta*B"})
pd.set_option("display.width", 220)
print(df.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
