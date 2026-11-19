"""G3d - which constructions are STABLE: same sign in-sample over a file's full history (what the
overfitting gate scores) AND in the scored validation blocks (what the board scores)?

Measured problem: within each shipped file the oldest history has the opposite relationship to the
scored rows (r07). A fixed signal tuned to the validation blocks (S06) is then strongly negative
in-sample and fails the gate emulation (p = 1.0). A legitimate submission must carry a relationship
that holds in BOTH places - this screen finds those; it does not try to fool the gate.

For each construction S (no fitting):
  gate_IC / gate_SR : mean IC and net Sharpe (336h smoothing, 5bp) over the training slice of the
                      gate reference file (latest dev period, all rows but the last 800)
  val_IC / val_SR   : the same over the 29 dev validation blocks (official protocol, from r06 logic)
  stable            : sign(gate_IC) == sign(val_IC) and both |t| >= 2 (NW)

  python research/r09_stability_screen.py   -> reports/r09_stability_screen.md / .csv
"""
# %%
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from _common import REPORTS, md_table, setup
from lab.backtest import pnl, sharpe
from lab.metrics import ic_series, mean_se_nw
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


def smooth(S):
    w = 0.5 ** (np.arange(WIN) / HL)
    A = S.fillna(0.0).to_numpy(float)
    return pd.DataFrame(lfilter(w, [1.0], A, axis=0) / lfilter(w, [1.0], np.ones((len(A), 1)), axis=0),
                        index=S.index, columns=S.columns)


import sys as _sys
FAMILY = "symmetric" if "--symmetric" in _sys.argv else "basic"


def constructions_symmetric(F):
    """Sign-symmetric (even) constructions: magnitudes and products - immune to selection-induced
    directional bias by construction."""
    tick = list(dict.fromkeys(F.columns.get_level_values(-1)))
    names = list(dict.fromkeys(F.columns.get_level_values(0)))
    raw = {n: F[n].reindex(columns=tick).astype(float) for n in names}
    zz = {n: z(raw[n]) for n in names}
    c = {}
    for n in names:
        c[f"rank |z {n}|"] = rank(zz[n].abs())
        for k in (1, 6, 24, 120):
            c[f"rank |z d{k} {n}|"] = rank(z(raw[n].diff(k)).abs())
    for a, b in itertools.combinations(names, 2):
        c[f"rank z{a}*z{b}"] = rank(zz[a] * zz[b])
        c[f"rank |z{a}-z{b}|"] = rank((zz[a] - zz[b]).abs())
    for n in names:
        for k in (24, 120):
            c[f"rank z{n}*zd{k}{n}"] = rank(zz[n] * z(raw[n].diff(k)))
    return c


def constructions(F):
    if FAMILY == "symmetric":
        return constructions_symmetric(F)
    return constructions_basic(F)


def constructions_basic(F):
    tick = list(dict.fromkeys(F.columns.get_level_values(-1)))
    names = list(dict.fromkeys(F.columns.get_level_values(0)))
    raw = {n: F[n].reindex(columns=tick).astype(float) for n in names}
    zz = {n: z(raw[n]) for n in names}
    c = {}
    for n in names:
        c[f"rank {n}"] = rank(raw[n])
        for k in (1, 6, 24, 120, 240):
            c[f"rank d{k} {n}"] = rank(raw[n].diff(k))
    for a, b in itertools.combinations(names, 2):
        c[f"rank z{a}-z{b}"] = rank(zz[a] - zz[b])
        c[f"rank z{a}*z{b}"] = rank(zz[a] * zz[b])
    return c


def score(S, T, X, idx):
    S = S.fillna(0.0)
    rho = ic_series(S, T, 1).reindex(idx).dropna()
    Sm = smooth(S)
    return rho, pnl(Sm.loc[idx], X.loc[idx], 5.0)


# gate reference: training slice of the latest dev period file
ref = dev_p[-1]
F = pd.read_parquet(root / f"{ref}.small.features.parquet")
T = pd.read_parquet(root / f"{ref}.small.target.parquet")
X = pd.read_parquet(root / f"{ref}.small.returns.parquet")
tr = F.index[: len(F) - 800]
gate = {}
for k, S in constructions(F).items():
    rho, pl = score(S, T, X, tr)
    ic, se = mean_se_nw(rho.to_numpy(), lag)
    gate[k] = {"gate_IC": ic, "gate_t": ic / se if se else np.nan, "gate_SR": sharpe(pl["net"]),
               "gate_IC_recent5k": float(rho.iloc[len(rho) - 5000:].mean())}
print(f"gate reference {ref}: {len(tr)} rows", flush=True)

val_rho, val_pnl = {}, {}
for p in dev_p:
    F = pd.read_parquet(root / f"{p}.small.features.parquet")
    T = pd.read_parquet(root / f"{p}.small.target.parquet")
    X = pd.read_parquet(root / f"{p}.small.returns.parquet")
    n = len(F)
    lo = n - 800 - WIN - 300
    F, T, X = F.iloc[lo:], T.iloc[lo:], X.iloc[lo:]
    vi = F.index[len(F) - 800:]
    for k, S in constructions(F).items():
        rho, pl = score(S, T, X, vi)
        val_rho.setdefault(k, []).append(rho)
        val_pnl.setdefault(k, []).append(pl["net"])
    print(f"  [{p}]", end="", flush=True)
print()

rows = []
for k in gate:
    rho = pd.concat(val_rho[k])
    ic, se = mean_se_nw(rho.to_numpy(), lag)
    per = np.array([r.mean() for r in val_rho[k]])
    row = {"construction": k, **gate[k], "val_IC": ic, "val_t": ic / se if se else np.nan,
           "val_pos": float((per > 0).mean()), "val_SR": sharpe(pd.concat(val_pnl[k]))}
    row["stable"] = bool(np.sign(row["gate_IC"]) == np.sign(ic) and abs(row["gate_t"]) >= 2 and abs(row["val_t"]) >= 2)
    rows.append(row)
df = pd.DataFrame(rows)
df["min_abs_t"] = np.minimum(df["gate_t"].abs(), df["val_t"].abs()) * (np.sign(df["gate_IC"]) == np.sign(df["val_IC"]))
df = df.sort_values("min_abs_t", ascending=False)
suffix = "_symmetric" if FAMILY == "symmetric" else ""
df.to_csv(REPORTS / f"r09_stability_screen{suffix}.csv", index=False)
head = f"{len(df)} constructions; stable (same sign, |t|>=2 in both gate slice and validation): {int(df['stable'].sum())}"
(REPORTS / f"r09_stability_screen{suffix}.md").write_text("\n".join([f"# R09 stability screen ({tag})", "", head, "", md_table(df)]), encoding="utf-8")
print(head)
print(df.head(25).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
