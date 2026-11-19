"""C5 r02 - broad RECENCY screen (pre-declared: the 90 r16 constructions, raw + perp-to-occupied, 2 smoothings).

Scored ONLY on the 10 most recent periods (dev 022-029 + vault 030-031) = the local proxy for the board's hidden band.
Every construction x smoothing counts as a trial (the count is appended to the registry at the end).
Kept: proxy t >= 2.5 (NW), >= 8/10 periods positive, |cos| <= 0.35 with all four occupied legs (S06, S07, S08, S09),
proxy @10bp > 0. Sign is fixed by the construction's FULL-dev-history orientation in C4's r16 (not chosen on recent data).
"""
import itertools

import numpy as np
import pandas as pd

from lab import registry
from lab.backtest import pnl, sharpe
from lab.honest import _load, blocks, occupied_raw, rank, residualize, smooth, unit, zx, leg_hl
from lab.metrics import row_cos, sharpe_se_nw
from lab.official import periods_split
from lab.paths import REPORTS, settings

s = settings()
devp, vp = periods_split(s)
PERIODS = devp[-8:] + vp
HLS = (48.0, 168.0)
r16 = pd.read_csv("../C4/reports/r16_x_alpha_screen.csv")
SIGN = dict(zip(r16["construction"], r16["sign"]))


def constructions(F):
    g = {f"Feature.{i}": v for i, v in blocks(F).items()}
    zz = {n: zx(g[n]) for n in g}
    c = {}
    for n in g:
        c[f"rank {n}"] = rank(g[n]); c[f"rank |z {n}|"] = rank(zz[n].abs())
        for k in (6, 24, 120, 240):
            c[f"rank d{k} {n}"] = rank(g[n].diff(k)); c[f"rank |z d{k} {n}|"] = rank(zx(g[n].diff(k)).abs())
    for a, b in itertools.combinations(g, 2):
        c[f"rank z{a}-z{b}"] = rank(zz[a] - zz[b]); c[f"rank z{a}*z{b}"] = rank(zz[a] * zz[b])
    occ = occupied_raw(F)
    out = {}
    for k, v in c.items():
        U = unit(v.fillna(0.0))
        out[k] = U
        out[f"{k} perp"] = unit(residualize(U, list(occ.values())))
    return out


acc = {}
for p in PERIODS:
    D = _load(p, 800 + 4 * 672 + 900)
    F, X = D["features"], D["returns"]
    vi = F.index[len(F) - 800:]
    legs = {k: smooth(v, leg_hl()[k]).loc[vi] for k, v in occupied_raw(F).items()}
    for k, U in constructions(F).items():
        sg = SIGN.get(k, 1)
        for hl in HLS:
            P = (sg * smooth(U, hl)).loc[vi]
            acc.setdefault((k, hl, 5), []).append(pnl(P, X.loc[vi], 5.0)["net"].reset_index(drop=True))
            acc.setdefault((k, hl, 10), []).append(pnl(P, X.loc[vi], 10.0)["net"].reset_index(drop=True))
            for lk, L in legs.items():
                acc.setdefault((k, hl, lk), []).append(row_cos(P, L).mean())
    print(f"  [{p}]", end="", flush=True)
print()
rows = []
for (k, hl, tag) in [key for key in acc if key[2] == 5]:
    sr5, se, _ = sharpe_se_nw(pd.concat(acc[(k, hl, 5)], ignore_index=True), 48)
    sr10, _, _ = sharpe_se_nw(pd.concat(acc[(k, hl, 10)], ignore_index=True), 48)
    row = {"construction": k, "hl": hl, "proxy5": sr5, "se": se, "t": sr5 / se if se else np.nan, "proxy10": sr10,
           "pos": float(np.mean([sharpe(x) > 0 for x in acc[(k, hl, 5)]]))}
    for lk in leg_hl():
        row[f"cos{lk[1:]}"] = float(np.mean(acc[(k, hl, lk)]))
    rows.append(row)
df = pd.DataFrame(rows)
cc = [c for c in df.columns if c.startswith("cos")]
df["max_cos"] = df[cc].abs().max(axis=1)
df["keep"] = (df["t"] >= 2.5) & (df["pos"] >= 0.8) & (df["max_cos"] <= 0.35) & (df["proxy10"] > 0)
df = df.sort_values("proxy5", ascending=False)
df.to_csv(REPORTS / "c5_r02_recency_screen.csv", index=False)
for _ in range(len(df)):
    registry.append({"run_id": "c5r02_screen_trial", "strategy": "c5_r02_recency_screen", "mode": "screen", "data": "real-official"})
print(f"{len(df)} trials; kept {int(df['keep'].sum())}")
print(df[df["keep"]].head(20).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
print("--- top 15 overall (any overlap):")
print(df.head(15)[["construction", "hl", "proxy5", "t", "proxy10", "pos", "max_cos"]].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
