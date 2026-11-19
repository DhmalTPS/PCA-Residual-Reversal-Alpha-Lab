"""G8 - regime-conditional stability of the two known edges (S06 reversal, S07 extremeness v2).

Purpose: a regime TIME-SHARE (hold S07 in regime A, S06 in regime B) is novel against both occupants if
each direction is held ~half the time - but it is legitimate (and valuable) only if the split is a real
regime in which each edge works better, and gate-safe only if the S06 leg is positive IN-SAMPLE over the
full in-file history in the regime where it is used. This screen measures exactly that.

Regime variables (same-hour cross-sectional state; scalar per hour) are compared with their own TRAILING
median (finite 720h window) -> high / low halves; causal by construction.
For each (signal, regime, half): mean IC vs target (NW t) on (a) the gate slice = training rows of the latest
dev file, (b) the 29 dev validation blocks; plus the share of hours in that half.

  python research/r14_regime_screen.py   -> reports/r14_regime_screen.md / .csv
"""
# %%
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import REPORTS, md_table, setup
from lab.metrics import ic_series, mean_se_nw
from lab.official import periods_split

s, _panel, tag = setup()
root = _panel.root
dev_p, _ = periods_split(s, root)
lag = int(s["nw_lag_hours"])
MED_WIN = 720


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


def build(F):
    tick = list(dict.fromkeys(F.columns.get_level_values(-1)))
    g = {i: F[f"Feature.{i}"].reindex(columns=tick).astype(float) for i in range(1, 7)}
    r = {i: rank(g[i]) for i in g}
    s06 = unit(-(r[1] + r[2] + r[3] + r[4]) / 4 + 0.5 * r[6] - 0.5 * r[5] - 0.5 * rank(g[1].diff(120)))
    s07 = unit(-sum(rank(z(g[i]).abs()) + rank(z(g[i].diff(24)).abs()) + rank(z(g[i].diff(120)).abs()) for i in (1, 2, 3, 4)))
    m1 = g[1].abs().mean(axis=1)
    reg = {
        "cv F1": g[1].std(axis=1) / m1.where(m1 > 0),
        "xs disp d24 F1": g[1].diff(24).std(axis=1),
        "xs disp d120 F1": g[1].diff(120).std(axis=1),
        "market trend d120 F1": g[1].diff(120).mean(axis=1),
        "|market trend| d120": g[1].diff(120).mean(axis=1).abs(),
        "xs mean F5": g[5].mean(axis=1),
        "xs disp F5": g[5].std(axis=1),
        "xs mean F6": g[6].mean(axis=1),
        "xs disp F6": g[6].std(axis=1),
    }
    hi = {k: (v > v.rolling(MED_WIN, min_periods=200).median()) for k, v in reg.items()}
    return {"S06": s06, "S07": s07}, hi


def cell_stats(rho, mask):
    x = rho[mask.reindex(rho.index).fillna(False).to_numpy()]
    if len(x) < 200:
        return np.nan, np.nan, len(x)
    ic, se = mean_se_nw(x.to_numpy(), lag)
    return ic, ic / se if se else np.nan, len(x)


# gate slice
ref = dev_p[-1]
F = pd.read_parquet(root / f"{ref}.small.features.parquet")
T = pd.read_parquet(root / f"{ref}.small.target.parquet")
tr = F.index[: len(F) - 800]
sig, hi = build(F)
gate = {}
for sn, S in sig.items():
    rho = ic_series(S, T, 1).reindex(tr).dropna()
    for rn, h in hi.items():
        for half, m in (("high", h), ("low", ~h)):
            gate[(sn, rn, half)] = cell_stats(rho, m.reindex(rho.index).fillna(False))

# validation blocks
val = {}
for p in dev_p:
    F = pd.read_parquet(root / f"{p}.small.features.parquet")
    T = pd.read_parquet(root / f"{p}.small.target.parquet")
    n = len(F)
    lo = n - 800 - MED_WIN - 300
    F, T = F.iloc[lo:], T.iloc[lo:]
    vi = F.index[len(F) - 800:]
    sig, hi = build(F)
    for sn, S in sig.items():
        rho = ic_series(S, T, 1).reindex(vi).dropna()
        for rn, h in hi.items():
            hh = h.reindex(rho.index).fillna(False)
            for half, m in (("high", hh), ("low", ~hh)):
                x = rho[m.to_numpy()]
                x.index = [f"{p}:{i}" for i in x.index]
                val.setdefault((sn, rn, half), []).append(x)
    print(f"  [{p}]", end="", flush=True)
print()

rows = []
for key, (gi, gt, gn) in gate.items():
    v = pd.concat(val[key]) if val.get(key) else pd.Series(dtype=float)
    if len(v) > 200:
        vic, vse = mean_se_nw(v.to_numpy(), lag)
        vt = vic / vse if vse else np.nan
    else:
        vic = vt = np.nan
    tot = sum(len(x) for k2, xs in val.items() if k2[:2] == key[:2] for x in xs)
    rows.append({"signal": key[0], "regime": key[1], "half": key[2], "gate_IC": gi, "gate_t": gt,
                 "val_IC": vic, "val_t": vt, "val_share": len(v) / tot if tot else np.nan})
df = pd.DataFrame(rows)
df["consistent_pos"] = (df["gate_t"] >= 1.5) & (df["val_t"] >= 2.0)
df.to_csv(REPORTS / "r14_regime_screen.csv", index=False)
(REPORTS / "r14_regime_screen.md").write_text("\n".join([f"# R14 regime-conditional stability ({tag})", "",
    "consistent_pos: gate-slice t >= 1.5 AND validation t >= 2.0 (the cells a gate-safe time-share could use).", "",
    md_table(df)]), encoding="utf-8")
pd.set_option("display.width", 220)
print(df.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
