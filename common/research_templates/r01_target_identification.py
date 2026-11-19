"""G1 - identify the target: alignment (offset), horizon h, how fast it decorrelates.

Uses look-ahead deliberately (lab.labels) - this is research, never submission code.

  python research/r01_target_identification.py           -> reports/r01_target.md
  python research/r01_target_identification.py --write   -> also writes config/derived.json
                                                           (target_lag, target_horizon_hours, embargo_hours)
                                                           ONLY allowed on the real data.
"""
# %%
from __future__ import annotations

import argparse
import json

import numpy as np

from _common import REPORTS, md_table, setup
from lab.labels import horizon_scan, target_autocorr
from lab.paths import CONFIG

ap = argparse.ArgumentParser()
ap.add_argument("--write", action="store_true")
args, _ = ap.parse_known_args()

s, panel, tag = setup()
X, T, D = panel.returns, panel.target, panel.dispersion

# %% horizon / offset scan
scan = horizon_scan(X, T, D)
best = scan.sort_values("cos_mean", ascending=False).iloc[0]
h, off = int(best["h"]), int(best["offset"])
# robust choice: smallest h within 95% of the best fit (horizon estimates are flat near the top)
near = scan[(scan["offset"] == off) & (scan["cos_mean"] >= 0.95 * best["cos_mean"])]
h_robust = int(near["h"].min())

# %% target persistence = ceiling on how slowly a perfect forecast's edge could decay
ac = target_autocorr(T)

# alignment: the scorer pairs P(i-1) with T(i) (IC) and with X(i) (PnL), and P(i-1) is computed from
# features up to row i-1. So training must pair features(i-1) with target(i): LAG = 1, whatever the
# offset. The offset only says which returns T(i) covers; offset >= 0 confirms T(i) starts no earlier
# than X(i), i.e. after everything P(i-1) can see (no leakage in that pairing).
lag = 1
if off < 0:
    print("WARNING: target window starts before X(i) - re-check alignment before trusting any result.")
embargo = int(h_robust + max(lag, 1) + 6)

lines = [f"# R01 target identification ({tag})", "",
         "cos_mean = mean per-hour cosine between target T(i) and sum_{k=off}^{off+h-1} X(i+k).",
         "disp_corr / disp_ratio: agreement of the shipped sigma(i) with the xs std of that window sum.", "",
         md_table(scan),
         "", f"**Best fit**: offset {off}, h = {h} (cos {best['cos_mean']:.3f}); robust h (95% of best) = {h_robust}",
         f"**Max cosine << 1** would mean the target is residualised / transformed beyond a plain window sum -> see r02.",
         "", "## Target autocorrelation  mean cos(T(i), T(i+k))", str({k: round(v, 3) for k, v in ac.items()}),
         "Overlapping windows make this ~ (h-k)/h for k < h; the observed decay bounds the alpha half-life from above.",
         "", f"## Derived: target_lag = {lag}, target_horizon_hours = {h_robust}, embargo_hours = {embargo}"]
out = REPORTS / "r01_target.md"
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

if args.write:
    if tag != "real":
        print("\nREFUSED --write on synthetic data (derived settings must come from the real target).")
    else:
        (CONFIG / "derived.json").write_text(json.dumps(
            {"target_lag": lag, "target_horizon_hours": h_robust, "embargo_hours": embargo,
             "_source": "research/r01_target_identification.py", "_best_cos": float(best["cos_mean"])}, indent=2))
        print(f"\nwrote {CONFIG / 'derived.json'}")
print(f"\n-> {out}")
