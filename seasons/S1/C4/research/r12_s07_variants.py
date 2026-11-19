"""G7 - pre-declared S07 improvement grid (8 variants, declared BEFORE looking; all logged as trials).

Motivation: S07 is blocked by a CURRENT-cycle signal (server global = quality-gate = 25.12 deg), most likely
board Q6 (SR 0.0075; IC/IC std/concentration match S07). Admission is greedy by Sharpe, so a higher-Sharpe
S07 would take that slot. Variants change only WHICH stable sign-symmetric members enter and the smoothing:
  members : L   = |z| levels F1-4
            LD  = L + |z| 24h changes F1-4                       (current S07)
            LDD = LD + |z| 120h changes F1-4                     (gate-stable, r09: t ~2.4)
            LDP = LD + products zF_a*zF5, a in F1-4               (stable family, r09)
  half-life : 336, 672
Selection: best by net SR @5bp; then the paired one-SE rule toward FEWER members / the current smoothing.
Every variant is evaluated under the official per-period protocol on dev periods only (vault is spent).

  python research/r12_s07_variants.py   -> reports/r12_s07_variants.md / .csv
"""
# %%
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from _common import REPORTS, md_table, setup
from lab import registry
from lab.backtest import pnl, sharpe
from lab.metrics import ic_series, mean_se_nw, sharpe_se_nw
from lab.official import periods_split

s, _panel, tag = setup()
root = _panel.root
dev_p, _ = periods_split(s, root)
lag = int(s["nw_lag_hours"])
HLS = (336.0, 672.0)


def rank(df):
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).to_numpy()[:, None]
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))


def z(df):
    d = df.sub(df.mean(axis=1), axis=0)
    return d.div(d.std(axis=1).replace(0, np.nan), axis=0)


def unit(S):
    d = S.sub(S.mean(axis=1), axis=0).fillna(0.0)
    n = np.sqrt((d * d).sum(axis=1))
    return d.div(n.where(n > 0), axis=0).fillna(0.0)


def smooth(S, hl):
    W = int(4 * hl)
    w = 0.5 ** (np.arange(W) / hl)
    A = S.fillna(0.0).to_numpy(float)
    out = pd.DataFrame(lfilter(w, [1.0], A, axis=0) / lfilter(w, [1.0], np.ones((len(A), 1)), axis=0),
                       index=S.index, columns=S.columns)
    return out.sub(out.mean(axis=1), axis=0)


def members(F):
    tick = list(dict.fromkeys(F.columns.get_level_values(-1)))
    g = {i: F[f"Feature.{i}"].reindex(columns=tick).astype(float) for i in range(1, 7)}
    L = [rank(z(g[i]).abs()) for i in (1, 2, 3, 4)]
    D = [rank(z(g[i].diff(24)).abs()) for i in (1, 2, 3, 4)]
    DD = [rank(z(g[i].diff(120)).abs()) for i in (1, 2, 3, 4)]
    PR = [rank(z(g[i]) * z(g[5])) for i in (1, 2, 3, 4)]
    return {"L": L, "LD": L + D, "LDD": L + D + DD, "LDP": L + D + PR}


acc = {}
for p in dev_p:
    F = pd.read_parquet(root / f"{p}.small.features.parquet")
    T = pd.read_parquet(root / f"{p}.small.target.parquet")
    X = pd.read_parquet(root / f"{p}.small.returns.parquet")
    n = len(F)
    lo = n - 800 - int(4 * max(HLS)) - 300
    F, T, X = F.iloc[lo:], T.iloc[lo:], X.iloc[lo:]
    vi = F.index[len(F) - 800:]
    for name, ms in members(F).items():
        U = unit(-sum(ms) / len(ms))
        acc.setdefault((name, "ic"), []).append(ic_series(U, T, 1).reindex(vi).dropna())
        for hl in HLS:
            Sm = smooth(U, hl)
            for c in (5.0, 10.0):
                acc.setdefault((name, hl, c), []).append(pnl(Sm.loc[vi], X.loc[vi], c)["net"])
    print(f"  [{p}]", end="", flush=True)
print()

rows, z5 = [], {}
for name in ("L", "LD", "LDD", "LDP"):
    rho = pd.concat(acc[(name, "ic")], ignore_index=True)
    ic, ise = mean_se_nw(rho.to_numpy(), lag)
    for hl in HLS:
        r5 = pd.concat(acc[(name, hl, 5.0)], ignore_index=True)
        r10 = pd.concat(acc[(name, hl, 10.0)], ignore_index=True)
        sr5, se5, _ = sharpe_se_nw(r5, lag)
        sr10, _, _ = sharpe_se_nw(r10, lag)
        per = np.array([sharpe(x) for x in acc[(name, hl, 5.0)]])
        z5[(name, hl)] = (r5 - r5.mean()) / r5.std() + sr5
        rows.append({"members": name, "halflife": hl, "IC": ic, "IC_t": ic / ise, "SR_5bp": sr5, "se": se5,
                     "SR_10bp": sr10, "periods_pos": float((per > 0).mean())})
        registry.append({"run_id": f"r12_{name}_{int(hl)}", "utc": pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%S"),
                         "strategy": "r12_s07_variants", "mode": "screen", "data": "real-official",
                         "IC": ic, "IC_t": ic / ise, "SR_net_5bp": sr5, "SR_se_5bp": se5, "SR_net_10bp": sr10,
                         "overrides": f"members={name};hl={hl}", "notes": "pre-declared S07 grid"})
df = pd.DataFrame(rows)
best = df.loc[df["SR_5bp"].idxmax()]
zb = z5[(best["members"], best["halflife"])]
df["se_diff_vs_best"] = [mean_se_nw((zb - z5[(r["members"], r["halflife"])]).to_numpy(), lag)[1] for _, r in df.iterrows()]
order = {"L": 0, "LD": 1, "LDP": 2, "LDD": 3}
cand = df[df["SR_5bp"] >= best["SR_5bp"] - df["se_diff_vs_best"]].copy()
cand["o"] = cand["members"].map(order) + (cand["halflife"] != 336.0) * 0.5
choice = cand.sort_values("o").iloc[0]
lines = [f"# R12 S07 pre-declared variants ({tag})", "", md_table(df), "",
         f"best @5bp: {best['members']} hl {best['halflife']} SR {best['SR_5bp']:+.4f}",
         f"chosen (paired one-SE, fewest members / current smoothing): {choice['members']} hl {choice['halflife']} "
         f"SR@5bp {choice['SR_5bp']:+.4f}, SR@10bp {choice['SR_10bp']:+.4f}", "8 trials logged (mode=screen)."]
(REPORTS / "r12_s07_variants.md").write_text("\n".join(lines), encoding="utf-8")
df.to_csv(REPORTS / "r12_s07_variants.csv", index=False)
print("\n".join(lines))
