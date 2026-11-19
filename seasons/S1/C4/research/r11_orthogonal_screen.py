"""G6 - what edge is left ORTHOGONAL to the occupied directions?

Server novelty checks (2026-10-07): S06 (level reversal) blocked at quality-gate 16.31 deg (rho ~0.96),
S07 (extremeness) at 25.12 deg (rho ~0.91). On the sphere, a candidate is >= 60 deg from both occupants
if it is >= 76 deg from S06 and >= 85 deg from S07. We guarantee that locally by projecting S06 and S07
out of every candidate HOUR BY HOUR (exact zero correlation with both), then ask whether any edge is left.

Per construction C (no fitting): residual R_t = C_t - proj_{span(S06_t, S07_t)} C_t, then
  val_IC / val_t / val_pos / val_SR : official per-period validation blocks (29 dev periods)
  gate_IC / gate_t                  : in-sample over the gate reference slice (training rows of file 029)
  rho06 / rho07                     : mean per-hour cosine of the SMOOTHED residual with smoothed S06/S07
                                      (smoothing re-introduces a little overlap; must stay small)
  python research/r11_orthogonal_screen.py   -> reports/r11_orthogonal_screen.md / .csv
"""
# %%
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from _common import REPORTS, md_table, setup
from lab.backtest import pnl, sharpe
from lab.metrics import ic_series, mean_se_nw, row_cos
from lab.official import periods_split

s, _panel, tag = setup()
root = _panel.root
dev_p, _ = periods_split(s, root)
lag = int(s["nw_lag_hours"])
HL, WIN = 336.0, 1344


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


def smooth(S):
    w = 0.5 ** (np.arange(WIN) / HL)
    A = S.fillna(0.0).to_numpy(float)
    out = pd.DataFrame(lfilter(w, [1.0], A, axis=0) / lfilter(w, [1.0], np.ones((len(A), 1)), axis=0),
                       index=S.index, columns=S.columns)
    return out.sub(out.mean(axis=1), axis=0)


def residualize(C, B):
    """Per-hour least-squares residual of C (T x J) on the columns of B (T x J x m), all de-meaned."""
    A = C.to_numpy(float)
    A = A - A.mean(axis=1, keepdims=True)
    Bm = B - B.mean(axis=1, keepdims=True)
    coef = np.einsum("tkj,tj->tk", np.linalg.pinv(Bm), A)
    return pd.DataFrame(A - np.einsum("tjk,tk->tj", Bm, coef), index=C.index, columns=C.columns)


def build(F):
    tick = list(dict.fromkeys(F.columns.get_level_values(-1)))
    names = list(dict.fromkeys(F.columns.get_level_values(0)))
    g = {n: F[n].reindex(columns=tick).astype(float) for n in names}
    zz = {n: z(g[n]) for n in names}
    r = {n: rank(g[n]) for n in names}
    s06 = unit(-(r["Feature.1"] + r["Feature.2"] + r["Feature.3"] + r["Feature.4"]) / 4 + 0.5 * r["Feature.6"]
               - 0.5 * r["Feature.5"] - 0.5 * rank(g["Feature.1"].diff(120)))
    s07 = unit(-sum(rank(zz[f"Feature.{i}"].abs()) + rank(z(g[f"Feature.{i}"].diff(24)).abs()) for i in (1, 2, 3, 4)) / 8)
    c = {}
    for n in names:
        c[f"rank {n}"] = r[n]
        c[f"rank |z {n}|"] = rank(zz[n].abs())
        for k in (1, 6, 24, 120):
            c[f"rank d{k} {n}"] = rank(g[n].diff(k))
            c[f"rank |z d{k} {n}|"] = rank(z(g[n].diff(k)).abs())
    for a, b in itertools.combinations(names, 2):
        c[f"rank z{a}-z{b}"] = rank(zz[a] - zz[b])
        c[f"rank z{a}*z{b}"] = rank(zz[a] * zz[b])
    for n in names:
        for k in (24, 120):
            c[f"rank z{n}*zd{k}{n}"] = rank(zz[n] * z(g[n].diff(k)))
    B = np.stack([s06.to_numpy(), s07.to_numpy()], axis=-1)
    res = {k: unit(residualize(v.fillna(0.0), B)) for k, v in c.items()}
    return res, s06, s07


# gate reference slice
ref = dev_p[-1]
F = pd.read_parquet(root / f"{ref}.small.features.parquet")
T = pd.read_parquet(root / f"{ref}.small.target.parquet")
tr = F.index[: len(F) - 800]
res, _, _ = build(F)
gate = {}
for k, R in res.items():
    rho = ic_series(R, T, 1).reindex(tr).dropna()
    ic, se = mean_se_nw(rho.to_numpy(), lag)
    gate[k] = (ic, ic / se if se else np.nan)
print(f"gate slice {ref}: {len(tr)} rows", flush=True)

acc = {}
for p in dev_p:
    F = pd.read_parquet(root / f"{p}.small.features.parquet")
    T = pd.read_parquet(root / f"{p}.small.target.parquet")
    X = pd.read_parquet(root / f"{p}.small.returns.parquet")
    n = len(F)
    lo = n - 800 - WIN - 300
    F, T, X = F.iloc[lo:], T.iloc[lo:], X.iloc[lo:]
    vi = F.index[len(F) - 800:]
    res, s06, s07 = build(F)
    m06, m07 = smooth(s06).loc[vi], smooth(s07).loc[vi]
    for k, R in res.items():
        Rm = smooth(R).loc[vi]
        acc.setdefault((k, "ic"), []).append(ic_series(R, T, 1).reindex(vi).dropna())
        acc.setdefault((k, "pnl"), []).append(pnl(Rm, X.loc[vi], 5.0)["net"])
        acc.setdefault((k, "r06"), []).append(row_cos(Rm, m06).mean())
        acc.setdefault((k, "r07"), []).append(row_cos(Rm, m07).mean())
    print(f"  [{p}]", end="", flush=True)
print()

rows = []
for k in gate:
    rho = pd.concat(acc[(k, "ic")], ignore_index=True)
    ic, se = mean_se_nw(rho.to_numpy(), lag)
    per = np.array([x.mean() for x in acc[(k, "ic")]])
    gi, gt = gate[k]
    rows.append({"construction": k, "val_IC": ic, "val_t": ic / se if se else np.nan, "val_pos": float((per > 0).mean()),
                 "val_SR_raw": sharpe(pd.concat(acc[(k, "pnl")], ignore_index=True)),
                 "gate_IC": gi, "gate_t": gt, "same_sign": bool(np.sign(gi) == np.sign(ic)),
                 "rho06": float(np.mean(acc[(k, "r06")])), "rho07": float(np.mean(acc[(k, "r07")]))})
df = pd.DataFrame(rows)
# orient every construction in its validation direction (flip = negate), report the oriented net SR
# needs a re-run of pnl for the flipped sign; approximate ranking by |val_t| and gate agreement here
df["score"] = np.minimum(df["val_t"].abs(), np.where(df["same_sign"], df["gate_t"].abs(), 0.0))
df = df.sort_values(["score", "val_t"], key=lambda c: c.abs() if c.name == "val_t" else c, ascending=False)
df.to_csv(REPORTS / "r11_orthogonal_screen.csv", index=False)
head = (f"{len(df)} constructions residualised on (S06, S07) per hour; "
        f"stable (same sign, both |t|>=2): {int(((df['val_t'].abs() >= 2) & (df['gate_t'].abs() >= 2) & df['same_sign']).sum())}")
(REPORTS / "r11_orthogonal_screen.md").write_text("\n".join([f"# R11 orthogonal screen ({tag})", "", head, "", md_table(df)]), encoding="utf-8")
print(head)
print(df.head(25).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
