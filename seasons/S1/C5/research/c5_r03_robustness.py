"""C5 r03 - robustness of A and B under a leaderboard-like test (PRE-DECLARED; neighbours inform confidence only).

Configs: A (L48 K8 hl168) + 6 one-step neighbours; B (L120 K3 hl48) + 6 neighbours; S09 as submitted (L120 K8 hl672) =
calibration point (board 0.0028). Scoring = vendor protocol (last 800 rows per period file, vendor backtest maths).
For every config and every period 001-031: net SR at 5 / 7.5 / 10 / 15 bp. Derived: full (001-029), recent proxy
(022-031), vault (030-031), a block bootstrap over the 10 recent PERIODS (2000 draws) of the pooled proxy SR.
"""
import numpy as np
import pandas as pd

from lab import registry
from lab.backtest import pnl, sharpe
from lab.honest import _load, blocks, smooth, unit
from lab.metrics import sharpe_se_nw
from lab.official import periods_split
from lab.paths import REPORTS, settings

ns = {}
exec(open("research/c5_r01_residual_family.py", encoding="utf-8").read().split("rows = []")[0], ns)
idio_rev = ns["idio_rev"]

CFG = {"A": (48, 8, 168.0), "A_L36": (36, 8, 168.0), "A_L72": (72, 8, 168.0), "A_K6": (48, 6, 168.0), "A_K10": (48, 10, 168.0),
       "A_hl120": (48, 8, 120.0), "A_hl240": (48, 8, 240.0),
       "B": (120, 3, 48.0), "B_L96": (96, 3, 48.0), "B_L144": (144, 3, 48.0), "B_K2": (120, 2, 48.0), "B_K4": (120, 4, 48.0),
       "B_hl36": (120, 3, 36.0), "B_hl72": (120, 3, 72.0),
       "S09_submitted": (120, 8, 672.0)}
COSTS = (5.0, 7.5, 10.0, 15.0)
s = settings()
devp, vp = periods_split(s)
allp = devp + vp
recent = devp[-8:] + vp
net = {(k, c): {} for k in CFG for c in COSTS}
for p in allp:
    D = _load(p, 800 + 4 * 672 + 900)
    F, X = D["features"], D["returns"]
    vi = F.index[len(F) - 800:]
    cache = {}
    for k, (L, K, hl) in CFG.items():
        if (L, K) not in cache:
            cache[(L, K)] = unit(idio_rev(F, 720, K, L))
        P = smooth(cache[(L, K)], hl).loc[vi]
        for c in COSTS:
            net[(k, c)][p] = pnl(P, X.loc[vi], c)["net"].reset_index(drop=True)
    print(f"[{p}]", end="", flush=True)
print()
rng = np.random.default_rng(20261015)
rows = []
for k in CFG:
    row = {"config": k}
    for c in COSTS:
        d = net[(k, c)]
        row[f"full@{c:g}"] = sharpe_se_nw(pd.concat([d[p] for p in devp], ignore_index=True), 48)[0]
        row[f"proxy@{c:g}"] = sharpe_se_nw(pd.concat([d[p] for p in recent], ignore_index=True), 48)[0]
        row[f"vault@{c:g}"] = sharpe_se_nw(pd.concat([d[p] for p in vp], ignore_index=True), 48)[0]
    d5 = net[(k, 5.0)]
    boots = []
    for _ in range(2000):
        pick = rng.choice(recent, size=len(recent), replace=True)
        boots.append(sharpe(pd.concat([d5[p] for p in pick], ignore_index=True)))
    boots = np.array(boots)
    row["proxy5_boot_sd"] = float(boots.std())
    row["proxy5_boot_p05"] = float(np.quantile(boots, 0.05))
    row["recent_periods_pos"] = float(np.mean([sharpe(d5[p]) > 0 for p in recent]))
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(REPORTS / "c5_r03_robustness.csv", index=False)
for k in CFG:
    if k not in ("A", "B", "S09_submitted"):
        registry.append({"run_id": f"c5r03_{k}", "strategy": "c5_r03_robustness", "mode": "screen", "data": "real-official",
                         "SR_net_5bp": float(df.loc[df.config == k, "full@5"].iloc[0]), "notes": "neighbourhood robustness (confidence only)"})
pd.set_option("display.width", 250)
cols = ["config", "full@5", "proxy@5", "proxy@7.5", "proxy@10", "proxy@15", "vault@5", "proxy5_boot_sd", "proxy5_boot_p05", "recent_periods_pos"]
print(df[cols].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
