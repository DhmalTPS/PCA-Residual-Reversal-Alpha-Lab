"""G1b - the "obvious span": textbook signals, their IC on the TARGET vs their PnL on X.

The rules say obvious signals score ~0 on the target by construction. This script measures it:
a signal with real PnL on X but ~0 IC on T reveals what the target was neutralised against.
Also reports pairwise overlap between the obvious signals and each strategy's OOS output, so
you know how much of a candidate is just a re-packaged obvious signal.

  python research/r02_obvious_span.py   -> reports/r02_obvious_span.md
"""
# %%
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import REPORTS, md_table, setup
from lab import registry
from lab.backtest import pnl
from lab.metrics import benjamini_hochberg, ic_series, mean_se_nw, p_two_sided, row_cos, sharpe_se_nw
from lab.paths import RUNS

s, panel, tag = setup()
F, X, T = panel.features, panel.returns, panel.target
lag = int(s["nw_lag_hours"])


def xs_z(df):
    d = df.sub(df.mean(axis=1), axis=0)
    return d.div(d.std(axis=1).replace(0, np.nan), axis=0).clip(-4, 4).fillna(0.0)


# %% candidate obvious signals (all causal)
obv = {}
for f in panel.feature_names:
    B = F[f].astype(float)
    obv[f"{f} level"] = xs_z(B)
    obv[f"{f} chg24"] = xs_z(B - B.rolling(24, min_periods=12).mean())
for w in (1, 6, 24, 96):
    obv[f"returns trailing {w}h (momentum)"] = xs_z(X.rolling(w, min_periods=1).sum())

rows = []
for name, S in obv.items():
    rho = ic_series(S, T, 1)
    ic, se = mean_se_nw(rho.to_numpy(), lag)
    sm = S.rolling(12, min_periods=1).mean()
    sr, sr_se, _ = sharpe_se_nw(pnl(sm, X, float(s["official_cost_bp"]))["net"], lag)
    gsr, _, _ = sharpe_se_nw(pnl(S, X, 0.0)["gross"], lag)
    rows.append({"signal": name, "IC_T": ic, "IC_t": ic / se if se else np.nan, "p": p_two_sided(ic / se) if se else np.nan,
                 "grossSR_X": gsr, "netSR_X_smoothed": sr, "SR_se": sr_se})
df = pd.DataFrame(rows)
df["BH_q10"] = benjamini_hochberg(df["p"].fillna(1).to_numpy(), 0.10)
df = df.sort_values("IC_t", key=np.abs, ascending=False)

# %% overlap of each strategy's latest OOS output with the obvious set
ov = []
for _, r in registry.latest_runs("full", tag).iterrows():
    P = pd.read_parquet(RUNS / r["run_id"] / "preds.parquet")
    cs = {k: float(row_cos(P, S).mean()) for k, S in obv.items()}
    top = sorted(cs.items(), key=lambda kv: -abs(kv[1]))[:3]
    ov.append({"strategy": r["strategy"], "top_overlaps": "; ".join(f"{k} {v:+.2f}" for k, v in top)})

lines = [f"# R02 obvious span ({tag})", "",
         "IC_T: mean per-hour IC vs the TARGET (NW t). grossSR_X / netSR_X: PnL on the returns panel (net uses 12h smoothing, official cost).",
         "BH_q10: survives Benjamini-Hochberg at q=10% across this family.",
         "**Signals with |IC_t| small but |grossSR_X| material = what the target neutralises. Orthogonalise candidates against them.**", "",
         md_table(df), "", "## Strategy overlap with the obvious set (mean per-hour cosine)",
         md_table(pd.DataFrame(ov)) if ov else "(no full runs yet)"]
out = REPORTS / "r02_obvious_span.md"
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print(f"\n-> {out}")
