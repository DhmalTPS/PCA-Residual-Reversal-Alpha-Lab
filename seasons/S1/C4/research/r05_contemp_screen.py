"""G3b - screen of SAME-HOUR cross-sectional constructions (the gate-friendly family).

Why this family: the vendor's demo_engineered.py documents (and our emulator reproduces) that
long rolling-window features fit the serially-correlated 120h target spuriously and fail the
overfitting gate, while contemporaneous cross-sectional ranks pass. Short changes (1h, 6h) are
included as near-contemporaneous.

For every construction S (xs rank in [-1,1]):
  IC_T  : mean per-hour IC vs the target (NW t), BH-FDR across the whole family
  SR_X  : Sharpe on the returns panel, gross and net @5bp after 48h finite smoothing
  stab  : share of P-blocks (one per period file) whose mean IC has the full-sample sign
Sign is reported raw (a negative IC means flip the construction).

  python research/r05_contemp_screen.py          -> reports/r05_contemp_screen.md / .csv
"""
# %%
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from _common import REPORTS, md_table, setup
from lab.backtest import pnl, sharpe
from lab.metrics import benjamini_hochberg, ic_series, mean_se_nw, p_two_sided

s, panel, tag = setup()
F, X, T = panel.features, panel.returns, panel.target
names = panel.feature_names
tick = panel.tickers
lag = int(s["nw_lag_hours"])
blocks = panel.months()
pmask = blocks.str.startswith("P")


def rank(df):
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).to_numpy()[:, None]
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))


def z(df):
    d = df.sub(df.mean(axis=1), axis=0)
    return d.div(d.std(axis=1).replace(0, np.nan), axis=0)


def smooth(S, w=48, hl=24.0):
    acc = 0.0
    tot = 0.0
    for k in range(w):
        wt = 0.5 ** (k / hl)
        acc = acc + S.shift(k).fillna(0.0) * wt
        tot += wt
    return acc / tot


raw = {n: F[n].reindex(columns=tick).astype(float) for n in names}
zz = {n: z(raw[n]) for n in names}
cands = {}
for n in names:
    cands[f"rank {n}"] = rank(raw[n])
    cands[f"rank d1 {n}"] = rank(raw[n].diff())
    cands[f"rank d6 {n}"] = rank(raw[n].diff(6))
for a, b in itertools.combinations(names, 2):
    cands[f"rank z{a}-z{b}"] = rank(zz[a] - zz[b])
    cands[f"rank z{a}*z{b}"] = rank(zz[a] * zz[b])

rows = []
for k, S in cands.items():
    S = S.fillna(0.0)
    rho = ic_series(S, T, 1)
    ic, se = mean_se_nw(rho.to_numpy(), lag)
    bym = rho.groupby(blocks.reindex(rho.index)).mean()
    bym = bym[bym.index.str.startswith("P")]
    Sm = smooth(S)
    rows.append({"construction": k, "IC_T": ic, "t": ic / se if se else np.nan,
                 "p": p_two_sided(ic / se) if se else np.nan,
                 "stab_P": float((np.sign(bym) == np.sign(ic)).mean()) if len(bym) else np.nan,
                 "SR_X_gross": sharpe(pnl(S, X, 0.0)["gross"]),
                 "SR_X_net_sm48": sharpe(pnl(Sm, X, 5.0)["net"]),
                 "SR_X_net_sm48_P": sharpe(pnl(Sm.loc[pmask], X.loc[pmask], 5.0)["net"])})
df = pd.DataFrame(rows)
df["BH_q10"] = benjamini_hochberg(df["p"].fillna(1).to_numpy(), 0.10)
df["abs_t"] = df["t"].abs()
df = df.sort_values("abs_t", ascending=False).drop(columns="abs_t")
df.to_csv(REPORTS / "r05_contemp_screen.csv", index=False)
lines = [f"# R05 same-hour cross-sectional screen ({tag})", "",
         f"{len(df)} constructions; {int(df['BH_q10'].sum())} survive BH q=10%. "
         "SR_X_* is the scored quantity (Sharpe on returns); *_P restricts to the 28 P-blocks (recent ~2.3y).",
         "Net uses a 48h finite exponential kernel (half-life 24h) and the official 5bp cost.", "", md_table(df)]
(REPORTS / "r05_contemp_screen.md").write_text("\n".join(lines), encoding="utf-8")
print(lines[2])
print(df.head(30).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
