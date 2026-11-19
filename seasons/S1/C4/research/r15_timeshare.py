"""G9 - regime time-share: S07 (extremeness v2) in HIGH cross-sectional dispersion, S06 (level reversal) in LOW.

Evidence (r14): S07's edge concentrates in high dispersion of 24h changes, in BOTH the gate slice (t 1.45 vs
0.31) and validation (t 4.2 vs 0.85); S06 works in both halves in validation but is toxic in-sample only in
HIGH dispersion (gate t -3.5 high vs +0.17 low). So: high -> S07, low -> S06.

Novelty arithmetic (server data: S07 v2 occupant cos 0.78, S06 pot occupant cos 0.96, S06 _|_ S07 locally):
rho_max ~ max(0.78*pi, 0.96*(1-pi)), minimised at pi ~ 0.55 -> rho ~ 0.43 (~64 deg). Each leg is smoothed
SEPARATELY and the switch is applied after smoothing, so every hour's direction is purely one leg.

Pre-declared grid (8 variants, logged as trials):
  regime variable : xs dispersion of 24h changes of F1 | of 120h changes of F1
  regime smoothing: raw | 168h trailing mean
  threshold       : trailing-720h median | trailing-720h 45th percentile (-> pi ~ 0.55)
Reported: val SR @5/10bp (official per-period), IC, pi, switches per 1000h, mean cos with each smoothed leg,
heuristic rho estimates vs the occupants, and the in-sample (gate-slice) net Sharpe.
"""
# %%
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from _common import REPORTS, md_table, setup
from lab import registry
from lab.backtest import pnl, sharpe
from lab.metrics import ic_series, mean_se_nw, row_cos, sharpe_se_nw
from lab.official import periods_split

s, _panel, tag = setup()
root = _panel.root
dev_p, _ = periods_split(s, root)
lag = int(s["nw_lag_hours"])
HL06, HL07, QWIN = 336.0, 672.0, 720


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


VARIANTS = list(itertools.product(("d24", "d120"), ("raw", "sm168"), (0.50, 0.45)))


def legs_and_gates(F):
    tick = list(dict.fromkeys(F.columns.get_level_values(-1)))
    g = {i: F[f"Feature.{i}"].reindex(columns=tick).astype(float) for i in range(1, 7)}
    r = {i: rank(g[i]) for i in g}
    s06 = unit(-(r[1] + r[2] + r[3] + r[4]) / 4 + 0.5 * r[6] - 0.5 * r[5] - 0.5 * rank(g[1].diff(120)))
    s07 = unit(-sum(rank(z(g[i]).abs()) + rank(z(g[i].diff(24)).abs()) + rank(z(g[i].diff(120)).abs()) for i in (1, 2, 3, 4)))
    A = unit(smooth(s07, HL07))          # S07 leg, unit after smoothing
    B = unit(smooth(s06, HL06))          # S06 leg
    gates = {}
    for var, rs, q in VARIANTS:
        d = g[1].diff(24 if var == "d24" else 120).std(axis=1)
        if rs == "sm168":
            d = d.rolling(168, min_periods=24).mean()
        thr = d.rolling(QWIN, min_periods=200).quantile(q)
        gates[(var, rs, q)] = (d > thr).astype(float)
    return A, B, gates


def combine(A, B, gt):
    P = A.mul(gt, axis=0) + B.mul(1.0 - gt, axis=0)
    return P.sub(P.mean(axis=1), axis=0)


# gate slice (in-sample)
ref = dev_p[-1]
F = pd.read_parquet(root / f"{ref}.small.features.parquet")
X = pd.read_parquet(root / f"{ref}.small.returns.parquet")
tr = F.index[: len(F) - 800]
A, B, gates = legs_and_gates(F)
gate_sr = {k: sharpe(pnl(combine(A, B, gt).loc[tr], X.loc[tr], 5.0)["net"]) for k, gt in gates.items()}
gate_sr["S07 only"] = sharpe(pnl(A.loc[tr], X.loc[tr], 5.0)["net"])
print(f"gate slice {ref}: {len(tr)} rows", flush=True)

acc = {}
for p in dev_p:
    F = pd.read_parquet(root / f"{p}.small.features.parquet")
    T = pd.read_parquet(root / f"{p}.small.target.parquet")
    X = pd.read_parquet(root / f"{p}.small.returns.parquet")
    n = len(F)
    lo = n - 800 - int(4 * HL07) - QWIN - 300
    F, T, X = F.iloc[lo:], T.iloc[lo:], X.iloc[lo:]
    vi = F.index[len(F) - 800:]
    A, B, gates = legs_and_gates(F)
    for k, gt in gates.items():
        P = combine(A, B, gt)
        for c in (5.0, 10.0):
            acc.setdefault((k, c), []).append(pnl(P.loc[vi], X.loc[vi], c)["net"].reset_index(drop=True))
        acc.setdefault((k, "ic"), []).append(ic_series(P, T, 1).reindex(vi).dropna().reset_index(drop=True))
        acc.setdefault((k, "pi"), []).append(float(gt.loc[vi].mean()))
        acc.setdefault((k, "sw"), []).append(float(gt.loc[vi].diff().abs().sum()) / len(vi) * 1000)
        acc.setdefault((k, "cosA"), []).append(row_cos(P.loc[vi], A.loc[vi]).mean())
        acc.setdefault((k, "cosB"), []).append(row_cos(P.loc[vi], B.loc[vi]).mean())
    print(f"  [{p}]", end="", flush=True)
print()

rows = []
for k in VARIANTS:
    sr5, se5, _ = sharpe_se_nw(pd.concat(acc[(k, 5.0)], ignore_index=True), lag)
    sr10, _, _ = sharpe_se_nw(pd.concat(acc[(k, 10.0)], ignore_index=True), lag)
    rho = pd.concat(acc[(k, "ic")], ignore_index=True)
    ic, ise = mean_se_nw(rho.to_numpy(), lag)
    cA, cB = float(np.mean(acc[(k, "cosA")])), float(np.mean(acc[(k, "cosB")]))
    row = {"regime": k[0], "reg_smooth": k[1], "q": k[2], "SR_5bp": sr5, "se": se5, "SR_10bp": sr10, "IC": ic, "IC_t": ic / ise,
           "pi_S07": float(np.mean(acc[(k, "pi")])), "switches_per_1000h": float(np.mean(acc[(k, "sw")])),
           "cos_S07leg": cA, "cos_S06leg": cB, "est_rho_S07occ": 0.78 * cA, "est_rho_S06occ": 0.96 * cB,
           "gate_inSR": gate_sr[k],
           "periods_pos": float(np.mean([sharpe(x) > 0 for x in acc[(k, 5.0)]]))}
    rows.append(row)
    registry.append({"run_id": f"r15_{k[0]}_{k[1]}_{k[2]}", "utc": pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%S"),
                     "strategy": "r15_timeshare", "mode": "screen", "data": "real-official", "IC": ic, "IC_t": ic / ise,
                     "SR_net_5bp": sr5, "SR_se_5bp": se5, "SR_net_10bp": sr10, "overrides": str(k), "notes": "pre-declared time-share grid"})
df = pd.DataFrame(rows)
df["est_rho_max"] = df[["est_rho_S07occ", "est_rho_S06occ"]].max(axis=1)
df.to_csv(REPORTS / "r15_timeshare.csv", index=False)
lines = [f"# R15 regime time-share S07(high disp) / S06(low disp) ({tag})", "",
         f"gate-slice in-sample net SR of S07 alone (reference): {gate_sr['S07 only']:+.4f}; fixed-signal null mean ~ -0.005.",
         "est_rho_* are heuristics: local cos with each leg x server cos of that leg with its occupant.", "", md_table(df)]
(REPORTS / "r15_timeshare.md").write_text("\n".join(lines), encoding="utf-8")
pd.set_option("display.width", 260)
print(lines[2])
print(df.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
