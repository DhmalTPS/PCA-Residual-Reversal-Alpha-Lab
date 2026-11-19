"""Skeptic #1 on peer_pca_K8_W720.
PLAN (evaluate() budget 4, using 3):
  E1 evaluate(fn, hl=672, warm=3888)  -> confirm reported metrics
  E2 evaluate(fn, hl=336, warm=3888)  -> smoothing robustness hl/2
  E3 evaluate(fn, hl=1344)            -> smoothing robustness 2*hl (default warm 5976)
Custom loop (no trial logging, diagnostics only): half-sample SRs (001-014 vs 015-029) at hl 336/672/1344,
cos to occupied legs smoothed at matching hl, cos to plain 120h F1 reversal, S08-residual variant SR.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from lab import honest as h
from lab.backtest import pnl, sharpe
from lab.metrics import row_cos, sharpe_se_nw
from research.agents.peer_factor_final import pca_idio_rev


def fn(F):
    return pca_idio_rev(F, W=720, K=8, L=120)


def custom():
    HL = [336.0, 672.0, 1344.0]
    warm = 5976
    per = {hl: [] for hl in HL}; ser = {hl: [] for hl in HL}
    cosm = {hl: {k: [] for k in ("S06", "S07", "S08", "F1rev")} for hl in HL}
    perp = []
    rawcos = {k: [] for k in ("S06", "S07", "S08")}
    ps = h.dev_periods()
    for p in ps:
        D = h._load(p, h.VAL + warm)
        F, X = D["features"], D["returns"]
        vi = F.index[len(F) - h.VAL:]
        U = h.unit(fn(F).reindex(index=F.index, columns=X.columns))
        occ = h.occupied_raw(F)
        f1r = h.unit(-h.blocks(F)[1].diff(120).reindex(columns=X.columns))
        for k in rawcos:
            rawcos[k].append(float(row_cos(U.loc[vi], occ[k].reindex(index=vi, columns=X.columns)).mean()))
        for hl in HL:
            Sm = h.smooth(U, hl).loc[vi]
            r = pnl(Sm, X.loc[vi], 5.0)["net"].reset_index(drop=True)
            ser[hl].append(r); per[hl].append(sharpe(r))
            for k in ("S06", "S07", "S08"):
                L = h.smooth(occ[k].reindex(columns=X.columns), hl).loc[vi]
                cosm[hl][k].append(float(row_cos(Sm, L).mean()))
            cosm[hl]["F1rev"].append(float(row_cos(Sm, h.smooth(f1r, hl).loc[vi]).mean()))
        P = h.residualize(U, [occ["S08"].reindex(columns=X.columns)])
        Sp = h.smooth(h.unit(P), 672.0).loc[vi]
        perp.append(pnl(Sp, X.loc[vi], 5.0)["net"].reset_index(drop=True))
        print(p, round(per[672.0][-1], 4), flush=True)
    out = {}
    for hl in HL:
        a = pd.concat(ser[hl][:14], ignore_index=True); b = pd.concat(ser[hl][14:], ignore_index=True)
        out[hl] = {"H1": sharpe_se_nw(a, 48)[:2], "H2": sharpe_se_nw(b, 48)[:2],
                   "pos_H1": float(np.mean(np.array(per[hl][:14]) > 0)), "pos_H2": float(np.mean(np.array(per[hl][14:]) > 0)),
                   "cos": {k: float(np.mean(v)) for k, v in cosm[hl].items()},
                   "cos_max_period": {k: float(np.max(np.abs(v))) for k, v in cosm[hl].items()}}
    out["per_period_672"] = [round(x, 4) for x in per[672.0]]
    out["perp8_SR"] = sharpe_se_nw(pd.concat(perp, ignore_index=True), 48)[:2]
    out["rawcos_unsmoothed"] = {k: float(np.mean(v)) for k, v in rawcos.items()}
    print(json.dumps(out, default=float, indent=1))


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "custom":
        custom()
    elif what == "eval":
        hl = float(sys.argv[2]); warm = int(sys.argv[3]) if len(sys.argv) > 3 else None
        m = h.evaluate(fn, f"skeptic1_peer_pca_K8_W720_hl{int(hl)}", hl=hl, warm=warm, notes="skeptic1")
        print("ADMISSIBLE", h.admissible(m))
