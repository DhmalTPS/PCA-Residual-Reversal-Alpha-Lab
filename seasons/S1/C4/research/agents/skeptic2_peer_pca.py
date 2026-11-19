"""Skeptic #2 for peer_pca_K8_W720.  PLAN (<=4 evaluate calls; using 3):
  E1 evaluate(fn, hl=672, warm=3888)   - confirm reported metrics
  E2 evaluate(fn, hl=336, warm=2664)   - smoothing robustness hl/2
  E3 evaluate(fn, hl=1344, warm=6696)  - smoothing robustness 2*hl
  Custom (no trial logging): per-period net-5bp SR at hl 336/672/1344, half-sample 001-014 vs 015-029,
  cos to occupied legs (harness legs, and legs smoothed at the candidate's own hl, and raw unsmoothed).
"""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, ".")
from lab import honest as h
from lab.backtest import pnl, sharpe
from lab.metrics import sharpe_se_nw, row_cos
from lab.paths import settings
from research.agents.peer_factor_final import pca_idio_rev

def fn(F):
    return pca_idio_rev(F, W=720, K=8, L=120)

lag = int(settings()["nw_lag_hours"])
mode = sys.argv[1]
if mode == "custom":
    HLS = [336, 672, 1344]
    acc = {hl: [] for hl in HLS}; per = {hl: [] for hl in HLS}
    cos = {}
    ps = h.dev_periods()
    for p in ps:
        D = h._load(p, h.VAL + 6696)
        F, X = D["features"], D["returns"]
        vi = F.index[len(F) - h.VAL:]
        U = h.unit(fn(F).reindex(index=F.index, columns=X.columns))
        occ = h.occupied_raw(F)
        legs = h.occupied_smoothed(p, 3888)
        for hl in HLS:
            Sm = h.smooth(U, hl).loc[vi]
            r = pnl(Sm, X.loc[vi], 5.0)["net"].reset_index(drop=True)
            acc[hl].append(r); per[hl].append(sharpe(r))
            for k in ("S06", "S07", "S08"):
                c1 = float(row_cos(Sm, legs[k].reindex(index=Sm.index, columns=Sm.columns)).mean())
                L2 = h.smooth(occ[k].reindex(columns=X.columns), hl).loc[vi]
                c2 = float(row_cos(Sm, L2).mean())
                cos.setdefault((hl, k, "harnessleg"), []).append(c1)
                cos.setdefault((hl, k, "samehl"), []).append(c2)
        for k in ("S06", "S07", "S08"):
            c0 = float(row_cos(U.loc[vi], occ[k].reindex(index=vi, columns=X.columns)).mean())
            cos.setdefault((0, k, "raw"), []).append(c0)
        print(p, {hl: round(per[hl][-1], 4) for hl in HLS}, flush=True)
    for hl in HLS:
        n = len(ps); a = pd.concat(acc[hl][:14], ignore_index=True); b = pd.concat(acc[hl][14:], ignore_index=True)
        sa = sharpe_se_nw(a, lag); sb = sharpe_se_nw(b, lag); sall = sharpe_se_nw(pd.concat(acc[hl], ignore_index=True), lag)
        print(f"hl={hl} all SR={sall[0]:.4f} se={sall[1]:.4f} | H1(001-014) SR={sa[0]:.4f} se={sa[1]:.4f} | "
              f"H2(015-029) SR={sb[0]:.4f} se={sb[1]:.4f} | pos={np.mean(np.array(per[hl])>0):.3f} "
              f"pos_H1={np.mean(np.array(per[hl][:14])>0):.3f} pos_H2={np.mean(np.array(per[hl][14:])>0):.3f}")
        print("   per-period:", [round(x, 3) for x in per[hl]])
    for key, v in sorted(cos.items()):
        print("cos", key, f"mean={np.mean(v):.4f} max={np.max(v):.4f} min={np.min(v):.4f}")
else:
    hl = float(mode); warm = int(4 * hl + 600 + 720)
    m = h.evaluate(fn, f"skeptic2_peer_pca_K8_W720_hl{int(hl)}", hl=hl, warm=warm, notes="skeptic2 verification")
    print("ADMISSIBLE", h.admissible(m))
