"""G3 - univariate screen of a strategy's OWN feature set (single source of truth: the strategy class).

For every feature block the strategy builds: mean IC vs target (NW t), BH-FDR across the family,
sign stability by month, IC decay over lags, and IC split by REALISED dispersion terciles
(diagnostic only - the realised dispersion is forward-looking and never a model input).

  python research/r03_feature_screen.py strategies/s03_peer_residual.py
"""
# %%
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import REPORTS, md_table, setup
from lab.io import month_labels
from lab.loader import load_strategy
from lab.metrics import benjamini_hochberg, ic_series, mean_se_nw, p_two_sided

ap = argparse.ArgumentParser()
ap.add_argument("file", nargs="?", default="strategies/s01_ridge_xs.py")
args, _ = ap.parse_known_args()
s, panel, tag = setup()
cls = load_strategy(args.file)
m = cls()
for meth in ("_peer_features", "_all_features", "_features"):
    if hasattr(m, meth):
        feats, _ = getattr(m, meth)(panel.features)
        break
lag = int(s["nw_lag_hours"])
months = panel.months()
disp = panel.dispersion

rows = []
for name, B in feats.items():
    rho = ic_series(B, panel.target, 1)
    if len(rho) < 100:
        continue
    ic, se = mean_se_nw(rho.to_numpy(), lag)
    bym = rho.groupby(months.reindex(rho.index)).mean()
    row = {"feature": name, "IC": ic, "t": ic / se if se else np.nan, "p": p_two_sided(ic / se) if se else np.nan,
           "months_same_sign": float((np.sign(bym) == np.sign(ic)).mean()),
           "IC_lag6": float(ic_series(B, panel.target, 6).mean()), "IC_lag24": float(ic_series(B, panel.target, 24).mean())}
    if disp is not None:
        d = disp.reindex(rho.index)
        q = d.quantile([1 / 3, 2 / 3]).to_numpy()
        row["IC_lowDisp"] = float(rho[d <= q[0]].mean())
        row["IC_highDisp"] = float(rho[d >= q[1]].mean())
    rows.append(row)
df = pd.DataFrame(rows)
df["BH_q10"] = benjamini_hochberg(df["p"].fillna(1).to_numpy(), 0.10)
df = df.sort_values("t", key=np.abs, ascending=False)

lines = [f"# R03 feature screen: {Path(args.file).stem} ({tag})", "",
         f"{len(df)} feature blocks; {int(df['BH_q10'].sum())} survive BH at q=10%. Univariate IC is a screen, not a verdict: "
         "interactions can matter where marginals are ~0 (that is the stated character of this target).",
         "months_same_sign = share of months whose mean IC has the full-sample sign (stability).",
         "IC_lowDisp / IC_highDisp use the REALISED forward dispersion: diagnostic only.", "", md_table(df)]
out = REPORTS / f"r03_screen_{Path(args.file).stem}.md"
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:6]))
print(df.head(20).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
print(f"\n-> {out}")
