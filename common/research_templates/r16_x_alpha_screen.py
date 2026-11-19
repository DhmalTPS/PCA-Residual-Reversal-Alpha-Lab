"""G10 - screen by what the BOARD scores: honest net Sharpe on the returns panel X (not IC on the target T).

Rationale: the target is neutralised (cos 0.936 with the raw window sum, r01) and most competitors fit T, so
constructions that pay on X without predicting T are the least likely to be occupied. Example: |z| of 120h
changes has IC_T ~ 0 (t 0.6) but net SR on X +0.0106 (r13).

For each construction C (raw, and residualised per hour on the S06/S07 directions = "perp"):
  orientation   : sign of its honest GROSS Sharpe on X over the 29 dev validation blocks
  SR5 / SR10    : honest net Sharpe @5/10bp after 672h smoothing (oriented)
  gate_inSR     : in-sample net Sharpe over the gate slice (training rows of the latest dev file), oriented
  cos06 / cos07 : mean per-hour cosine of the smoothed oriented signal with the smoothed S06 / S07-v2 legs
  IC_T          : oriented IC vs the target (to show which ones are 'X-only')
Candidates = SR5 >= 0.008, gate_inSR >= -0.003 (above the fixed-signal null q95), |cos06|,|cos07| <= 0.35.
  python research/r16_x_alpha_screen.py   -> reports/r16_x_alpha_screen.md / .csv
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
HL = 672.0


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


def smooth(S, hl=HL):
    W = int(4 * hl)
    w = 0.5 ** (np.arange(W) / hl)
    A = S.fillna(0.0).to_numpy(float)
    out = pd.DataFrame(lfilter(w, [1.0], A, axis=0) / lfilter(w, [1.0], np.ones((len(A), 1)), axis=0),
                       index=S.index, columns=S.columns)
    return out.sub(out.mean(axis=1), axis=0)


def residualize(C, Bs):
    A = C.to_numpy(float)
    A = A - A.mean(axis=1, keepdims=True)
    B = np.stack([b.to_numpy(float) for b in Bs], axis=-1)
    B = B - B.mean(axis=1, keepdims=True)
    coef = np.einsum("tkj,tj->tk", np.linalg.pinv(B), A)
    return pd.DataFrame(A - np.einsum("tjk,tk->tj", B, coef), index=C.index, columns=C.columns)


def build(F):
    tick = list(dict.fromkeys(F.columns.get_level_values(-1)))
    names = list(dict.fromkeys(F.columns.get_level_values(0)))
    g = {n: F[n].reindex(columns=tick).astype(float) for n in names}
    zz = {n: z(g[n]) for n in names}
    r = {n: rank(g[n]) for n in names}
    s06 = unit(-(r["Feature.1"] + r["Feature.2"] + r["Feature.3"] + r["Feature.4"]) / 4 + 0.5 * r["Feature.6"]
               - 0.5 * r["Feature.5"] - 0.5 * rank(g["Feature.1"].diff(120)))
    s07 = unit(-sum(rank(zz[f"Feature.{i}"].abs()) + rank(z(g[f"Feature.{i}"].diff(24)).abs())
                    + rank(z(g[f"Feature.{i}"].diff(120)).abs()) for i in (1, 2, 3, 4)))
    c = {}
    for n in names:
        c[f"rank {n}"] = r[n]
        c[f"rank |z {n}|"] = rank(zz[n].abs())
        for k in (6, 24, 120, 240):
            c[f"rank d{k} {n}"] = rank(g[n].diff(k))
            c[f"rank |z d{k} {n}|"] = rank(z(g[n].diff(k)).abs())
    for a, b in itertools.combinations(names, 2):
        c[f"rank z{a}-z{b}"] = rank(zz[a] - zz[b])
        c[f"rank z{a}*z{b}"] = rank(zz[a] * zz[b])
    out = {}
    for k, v in c.items():
        U = unit(v.fillna(0.0))
        out[f"{k}"] = U
        out[f"{k} perp"] = unit(residualize(U, [s06, s07]))
    return out, smooth(s06, 336.0), smooth(s07, HL)


# gate slice
ref = dev_p[-1]
F = pd.read_parquet(root / f"{ref}.small.features.parquet")
X = pd.read_parquet(root / f"{ref}.small.returns.parquet")
tr = F.index[: len(F) - 800]
cons, _, _ = build(F)
gate = {}
for k, U in cons.items():
    Sm = smooth(U)
    gate[k] = (sharpe(pnl(Sm.loc[tr], X.loc[tr], 5.0)["net"]), sharpe(pnl(-Sm.loc[tr], X.loc[tr], 5.0)["net"]))
print(f"gate slice {ref} done", flush=True)

acc = {}
for p in dev_p:
    F = pd.read_parquet(root / f"{p}.small.features.parquet")
    T = pd.read_parquet(root / f"{p}.small.target.parquet")
    X = pd.read_parquet(root / f"{p}.small.returns.parquet")
    n = len(F)
    lo = n - 800 - int(4 * HL) - 300
    F, T, X = F.iloc[lo:], T.iloc[lo:], X.iloc[lo:]
    vi = F.index[len(F) - 800:]
    cons, l06, l07 = build(F)
    for k, U in cons.items():
        Sm = smooth(U).loc[vi]
        Xv = X.loc[vi]
        acc.setdefault((k, "g"), []).append(pnl(Sm, Xv, 0.0)["gross"].reset_index(drop=True))
        for sign in (1, -1):
            for c in (5.0, 10.0):
                acc.setdefault((k, sign, c), []).append(pnl(sign * Sm, Xv, c)["net"].reset_index(drop=True))
        acc.setdefault((k, "ic"), []).append(ic_series(U, T, 1).reindex(vi).dropna().reset_index(drop=True))
        acc.setdefault((k, "c06"), []).append(row_cos(Sm, l06.loc[vi]).mean())
        acc.setdefault((k, "c07"), []).append(row_cos(Sm, l07.loc[vi]).mean())
    print(f"  [{p}]", end="", flush=True)
print()

rows = []
for k in gate:
    sign = 1 if sharpe(pd.concat(acc[(k, "g")], ignore_index=True)) >= 0 else -1
    r5 = pd.concat(acc[(k, sign, 5.0)], ignore_index=True)
    sr5, se5, _ = sharpe_se_nw(r5, lag)
    sr10, _, _ = sharpe_se_nw(pd.concat(acc[(k, sign, 10.0)], ignore_index=True), lag)
    rho = pd.concat(acc[(k, "ic")], ignore_index=True) * sign
    ic, ise = mean_se_nw(rho.to_numpy(), lag)
    rows.append({"construction": k, "sign": sign, "SR5": sr5, "se": se5, "SR10": sr10,
                 "periods_pos": float(np.mean([sharpe(x) > 0 for x in acc[(k, sign, 5.0)]])),
                 "gate_inSR": gate[k][0] if sign == 1 else gate[k][1], "IC_T": ic, "IC_T_t": ic / ise,
                 "cos06": sign * float(np.mean(acc[(k, "c06")])), "cos07": sign * float(np.mean(acc[(k, "c07")]))})
df = pd.DataFrame(rows)
df["candidate"] = (df["SR5"] >= 0.008) & (df["gate_inSR"] >= -0.003) & (df["cos06"].abs() <= 0.35) & (df["cos07"].abs() <= 0.35)
df = df.sort_values("SR5", ascending=False)
df.to_csv(REPORTS / "r16_x_alpha_screen.csv", index=False)
registry.append({"run_id": "r16_screen", "utc": pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%S"), "strategy": "r16_x_alpha_screen",
                 "mode": "screen", "data": "real-official", "notes": f"{len(df)} oriented constructions screened (count as trials)"})
for _ in range(len(df) - 1):
    registry.append({"run_id": "r16_screen_trial", "strategy": "r16_x_alpha_screen", "mode": "screen", "data": "real-official"})
head = f"{len(df)} oriented constructions; candidates: {int(df['candidate'].sum())}"
(REPORTS / "r16_x_alpha_screen.md").write_text("\n".join([f"# R16 X-alpha screen ({tag})", "", head, "", md_table(df)]), encoding="utf-8")
pd.set_option("display.width", 240)
print(head)
print(df.head(30).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
