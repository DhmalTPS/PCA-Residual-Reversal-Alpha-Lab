"""G4 - turnover control: smoothing half-life x no-trade band, chosen under STRESS cost with a one-SE rule.

1. One walk-forward with smoothing and band OFF -> raw out-of-sample unit-norm score.
2. Re-apply the strategy's OWN _smooth/_band (identical code path to the submission) over a grid.
3. For each grid point: net Sharpe at official and stress cost, NW s.e., turnover, gross.
4. Pick the best by stress-cost Sharpe, then the MOST smoothing within one s.e. of it
   (smoothing is the conservative direction: lower cost uncertainty, less churn on the hidden band).
Every grid point is logged as a trial (mode='grid') so the deflated Sharpe stays honest.

  python research/r04_turnover.py strategies/s02_lgbm_regime.py
"""
# %%
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import REPORTS, md_table, setup
from lab import registry
from lab.backtest import pnl, sharpe
from lab.loader import file_sha, load_strategy, with_overrides
from lab.metrics import mean_se_nw, sharpe_se_nw
from lab.official import fold_pnl, periods_split, run_official, val_only

ap = argparse.ArgumentParser()
ap.add_argument("file", nargs="?", default="strategies/s02_lgbm_regime.py")
ap.add_argument("--halflives", default="0,3,6,12,24,48")
ap.add_argument("--bands", default="0,0.25,0.5,1.0")
args, _ = ap.parse_known_args()
s, _dev_panel, tag = setup()
tag = tag + "-official"
base = load_strategy(args.file)
has_band = hasattr(base, "BAND") and hasattr(base, "_band")
raw_cls = with_overrides(base, {"SMOOTH_HALFLIFE": 0.0, **({"BAND": 0.0} if has_band else {})})
if not has_band:
    args.bands = "0"
hls = [float(x) for x in args.halflives.split(",")]
extra = int(max(4 * max(hls), 24)) + 48
print(f"official per-period folds, smoothing/band OFF, {extra} warm-up rows per fold ...")
dev_p, _ = periods_split(s)
folds = run_official(raw_cls, s, dev_p, extra_history=extra, log=lambda *_: None)
lag = int(s["nw_lag_hours"])
c0, c1 = float(s["official_cost_bp"]), float(s["stress_cost_bp"])
rows, zser = [], {}

for hl in hls:
    window = max(1, int(round(4 * hl))) if hl > 0 else 1
    for kappa in [float(x) for x in args.bands.split(",")]:
        over = {}
        for f in folds:
            Sm = base._smooth(f["P"], hl, window) if hl > 0 else f["P"]
            Pf = base._band(Sm, kappa, max(24, window)) if (kappa > 0 and has_band) else Sm
            over[f["period"]] = Pf.sub(Pf.mean(axis=1), axis=0)
        vf = val_only(folds, over)
        p0 = fold_pnl(vf, c0)
        p1 = fold_pnl(vf, c1)
        sr0, se0, _ = sharpe_se_nw(p0["net"], lag)
        sr1, se1, _ = sharpe_se_nw(p1["net"], lag)
        r1 = p1["net"]
        zser[(hl, kappa)] = (r1 - r1.mean()) / r1.std() + sr1
        rows.append({"halflife": hl, "window": window, "band": kappa, "SR_gross": sharpe(p0["gross"]),
                     "SR_5bp": sr0, "SR_10bp": sr1, "se": se1, "turnover": float(p0["turnover"].mean())})
df = pd.DataFrame(rows)

best = df.loc[df["SR_10bp"].idxmax()]
# One-SE rule on the PAIRED difference: configurations of one signal have highly correlated PnL,
# so the s.e. of SR_best - SR_c (NW on the per-hour difference of standardised returns) is far
# smaller than either standalone s.e.
zb = zser[(best["halflife"], best["band"])]
d_se = []
for _, r in df.iterrows():
    d = (zb - zser[(r["halflife"], r["band"])]).dropna().to_numpy()
    d_se.append(mean_se_nw(d, lag)[1] if len(d) > 10 else np.nan)
df["se_diff_vs_best"] = d_se
cand = df[df["SR_10bp"] >= best["SR_10bp"] - df["se_diff_vs_best"].fillna(0)]
choice = cand.sort_values(["halflife", "band"], ascending=False).iloc[0]
hl_grid = sorted(df["halflife"].unique())
bd_grid = sorted(df["band"].unique())
edge = [n for n, v, g in (("half-life", choice["halflife"], hl_grid), ("band", choice["band"], bd_grid)) if v == g[len(g) - 1]]

for _, r in df.iterrows():
    registry.append({"run_id": f"grid_{Path(args.file).stem}_{r['halflife']}_{r['band']}", "utc": pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S"),
                     "strategy": Path(args.file).stem, "file_sha": file_sha(args.file), "mode": "grid", "data": tag,
                     "overrides": json.dumps({"SMOOTH_HALFLIFE": r["halflife"], "SMOOTH_WINDOW": int(r["window"]), "BAND": r["band"]}),
                     "SR_net_5bp": r["SR_5bp"], "SR_net_10bp": r["SR_10bp"], "SR_gross": r["SR_gross"], "notes": "r04 turnover grid"})

flags = f"--set SMOOTH_HALFLIFE={choice['halflife']} --set SMOOTH_WINDOW={int(choice['window'])}"
if has_band:
    flags += f" --set BAND={choice['band']} --set BAND_WINDOW={max(24, int(choice['window']))}"
lines = [f"# R04 turnover: {Path(args.file).stem} ({tag})", "",
         "Grid over smoothing half-life (finite kernel, window = 4 x half-life) and no-trade band (fraction of the typical |position|).",
         f"Stress cost {c1:g} bp used for the choice; one-SE rule (paired-difference s.e.) toward MORE smoothing.", "",
         md_table(df.sort_values("SR_10bp", ascending=False)),
         "", f"**Best @10bp**: half-life {best['halflife']}, band {best['band']} -> SR {best['SR_10bp']:+.4f} (se {best['se']:.4f})",
         f"**Chosen (one-SE, most conservative)**: half-life {choice['halflife']}, band {choice['band']} -> SR@10bp {choice['SR_10bp']:+.4f}, SR@5bp {choice['SR_5bp']:+.4f}",
         (f"**WARNING: choice sits on the grid edge ({', '.join(edge)}) - extend the grid before trusting it.**" if edge else "Choice is interior to the grid."),
         f"Re-evaluate with: `python tools/evaluate.py {args.file} {flags}`",
         f"{len(df)} grid points were logged as trials."]
out = REPORTS / f"r04_turnover_{Path(args.file).stem}.md"
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print(f"\n-> {out}")
