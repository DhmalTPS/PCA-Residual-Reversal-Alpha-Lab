"""Family F5F6_NONLINEAR - round 2 (2026-10-08).

BUDGET: the family already spent 2 evaluate() calls in round 1 (f6_absmean336 SR +0.0068, f6_lvlstd336 SR -0.0086,
see F5F6_NONLINEAR_evals.jsonl). Counting them against the HARD LIMIT of 12 leaves 10 for this round.

KEY OBSERVATIONS (EDA, no scored rows touched):
  * F6 is piecewise constant on 8-hour blocks, ~1e-4 scale -> a perpetual FUNDING RATE. F5 is right-skewed,
    ac1 0.55 -> abnormal volume / activity. F1-F4 are near-identical trailing-return levels (ac1 0.97, ac24 0.36).
  * evaluate() smooths with hl 672 h (4 weeks): only SLOW asset characteristics survive the smoother. So the
    family is screened as slow characteristics built from nonlinear / conditional F5-F6 dynamics
    (funding vol / skew / persistence / price coupling, volume-return coupling, spike returns, information
    discreteness, lottery MAX, co-skewness...).
  * Round 1 lesson: screening on rows -5000..-900 mis-predicted (lvlstd flipped sign). Round 2 screens on THREE
    pre-scoring segments, A=-5000..-2400, B=-2400..-1600, C=-1600..-800 (the scored block is removed from the frame
    BEFORE anything is computed), and calibrates the proxy on the occupied legs + the two round-1 evals.

PLAN for evaluate() (max 10 this round):
  1-4 : best constructions with consistent sign in B and C (and A not strongly opposite), SR_C>0, max|cos|<=0.35
        (raw) - evaluated raw at hl 672.
  5-6 : for the best 2, the 'orth' version (smoothed in fn, residualised per hour on the three smoothed occupied
        legs; evaluate(hl=0)) to guarantee cos~0 - only if raw max|cos| > 0.25.
  7-8 : one pre-declared combination of evaluated survivors (equal-weight unit sum) if >=2 have SR>0.008.
  9-10: reserve; not used unless borderline.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.signal import lfilter

sys.path.insert(0, ".")
from lab.honest import (rank, zx, unit, smooth, residualize, blocks, occupied_raw, dev_periods, _load,  # noqa
                        evaluate, admissible, LEG_HL)
from lab.backtest import pnl, sharpe
from lab.metrics import ic_series, row_cos

HL = 672.0
SEGS = {"A": (5000, 2400), "B": (2400, 1600), "C": (1600, 800)}


def rm(df, w):
    return df.rolling(w, min_periods=w // 2).mean()


def rs(df, w):
    return df.rolling(w, min_periods=w // 2).std()


def rc(a, b, w):
    return a.rolling(w, min_periods=w // 2).corr(b)


def cands4(F):
    g = blocks(F)
    P = (g[1] + g[2] + g[3] + g[4]) / 4
    dP = P.diff(1)
    m = dP.mean(axis=1)
    M = pd.DataFrame(np.repeat(m.to_numpy()[:, None], dP.shape[1], 1), index=dP.index, columns=dP.columns)
    f5, f6 = g[5], g[6]
    d6 = f6.diff(8)
    c = {}
    c["CAL_f6_absmean336"] = rank(rm(f6.abs(), 336))
    c["CAL_f6_lvlstd336"] = rank(rs(f6, 336))
    c["f6_vol336"] = rank(rs(f6.diff(), 336))
    c["f6_negfrac336"] = rank(rm((f6 < 0).astype(float), 336))
    c["f6_skew672"] = rank(f6.rolling(672, min_periods=336).skew())
    c["f6_ar8_672"] = rank(rc(d6, d6.shift(8), 672))
    c["f6_dP_corr336"] = rank(rc(d6, P.diff(8), 336))
    c["f6lvl_P24_corr336"] = rank(rc(f6, P.diff(24), 336))
    c["f6_trend168_672"] = rank(rm(f6, 168) - rm(f6, 672))
    c["f6_excess336"] = rank(rm(residualize(rank(f6), [rank(P)]), 336))
    c["f6_absmean336_tails"] = (lambda r: r.where(r.abs() > 0.5, 0.0))(rank(rm(f6.abs(), 336)))
    c["f5_mean336"] = rank(rm(f5, 336))
    c["f5_med336"] = rank(f5.rolling(336, min_periods=168).median())
    c["f5_spikefreq336"] = rank(rm((f5 > 3).astype(float), 336))
    c["f5_skew672"] = rank(f5.rolling(672, min_periods=336).skew())
    c["f5_dP_corr336"] = rank(rc(f5, dP, 336))
    c["f5_absdP_corr336"] = rank(rc(f5, dP.abs(), 336))
    sp = (f5 > 2).astype(float)
    c["spike_ret336"] = rank((dP * sp).rolling(336, min_periods=168).sum() / sp.rolling(336, min_periods=168).sum().replace(0, np.nan))
    c["llorente336"] = rank(rc(dP, (f5 * dP).shift(1), 336))
    c["f5_f6_corr672"] = rank(rc(f5, d6, 672))
    c["dP_ac1_336"] = rank(rc(dP, dP.shift(1), 336))
    cov = (dP * M).rolling(672, min_periods=336).mean() - rm(dP, 672) * rm(M, 672)
    c["beta672"] = rank(cov / rs(M, 672) ** 2)
    c["idvol336"] = rank(rs(dP - M, 336))
    c["max24_336"] = rank(P.diff(24).rolling(336, min_periods=168).max())
    c["skew_dP672"] = rank(dP.rolling(672, min_periods=336).skew())
    Mc = M - rm(M, 672)
    c["coskew672"] = rank(rm((dP - rm(dP, 672)) * Mc * Mc, 672) / (rs(dP, 672) * rs(M, 672) ** 2))
    neg = (M < 0).astype(float)
    c["downcorr672"] = rank(rc(dP.where(neg > 0), M.where(neg > 0), 672))
    c["volratio72_672"] = rank(rs(dP, 72) / rs(dP, 672))
    pos = rm((dP > 0).astype(float), 672)
    c["infodisc672"] = rank(np.sign(P.diff(672)) * ((1 - pos) - pos))
    c["uphour_frac336"] = rank(rm((dP > 0).astype(float), 336))
    c["f6pos_x_rev"] = rank(-P.diff(120)) * (rank(f6) > 0.5)
    c["f6neg_x_rev"] = rank(-P.diff(120)) * (rank(f6) < -0.5)
    return c


def legs_raw(F):
    return {f"LEG_{k}": v for k, v in occupied_raw(F).items()}


def screen4():
    out = {}
    for p in dev_periods():
        D = _load(p, 5000 + 3400)
        n = len(D["features"])
        D = {k: v.iloc[: n - 800] for k, v in D.items()}  # the scored block is dropped BEFORE any computation
        F, X, T = D["features"], D["returns"], D["target"]
        n = len(F)
        legs = {k: v.reindex(columns=X.columns) for k, v in occupied_raw(F).items()}
        sl = {k: smooth(v, LEG_HL[k]) for k, v in legs.items()}
        cc = {**legs_raw(F), **cands4(F)}
        for name, C in cc.items():
            hl = LEG_HL[name[4:]] if name.startswith("LEG") else HL
            U = unit(C.reindex(columns=X.columns))
            S = smooth(U, hl)
            net = pnl(S, X, 5.0)["net"]
            rec = []
            for a, b in SEGS.values():
                idx = F.index[n - (a - 800): n - (b - 800)] if b > 800 else F.index[n - (a - 800):]
                rec.append(sharpe(net.reindex(idx)))
            idxC = F.index[n - 800:]
            rec.append(float(ic_series(U, T, 1).reindex(idxC).mean()))
            for k in ("S06", "S07", "S08"):
                rec.append(float(row_cos(S.loc[idxC], sl[k].loc[idxC]).mean()))
            out.setdefault(name, []).append(rec)
        print(p, flush=True)
    rows = []
    for name, v in out.items():
        a = np.array(v, float)
        r = dict(name=name)
        for j, s in enumerate(SEGS):
            x = a[:, j]
            r[f"SR{s}"] = np.nanmean(x)
            r[f"t{s}"] = np.nanmean(x) / np.nanstd(x) * np.sqrt(np.isfinite(x).sum())
            r[f"pos{s}"] = np.nanmean(x > 0)
        r["IC_C"] = np.nanmean(a[:, 3])
        r["c06"], r["c07"], r["c08"] = np.nanmean(a[:, 4]), np.nanmean(a[:, 5]), np.nanmean(a[:, 6])
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("SRC", ascending=False)
    df.to_csv("research/agents/F5F6_NONLINEAR_screen4.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(4).to_string())


if __name__ == "__main__" and sys.argv[1] == "screen4":
    screen4()


# ----------------------------------------------------------------------------- screen 5: orth versions + composites
SEL = ["CAL_f6_absmean336", "f6_vol336", "f5_skew672", "coskew672", "skew_dP672", "f6_excess336",
       "volratio72_672", "CAL_f6_lvlstd336", "max24_336"]
COMP = ["CAL_f6_absmean336", "f6_vol336", "f5_skew672", "coskew672"]


def smoothed_legs(F, cols):
    return {k: smooth(v.reindex(columns=cols), LEG_HL[k]) for k, v in occupied_raw(F).items()}


def orth(S, legs):
    return residualize(S, [legs[k] for k in ("S06", "S07", "S08")])


def screen5():
    out, geo = {}, {}
    for p in dev_periods():
        D = _load(p, 5000 + 3400)
        n = len(D["features"])
        D = {k: v.iloc[: n - 800] for k, v in D.items()}
        F, X = D["features"], D["returns"]
        n = len(F)
        sl = smoothed_legs(F, X.columns)
        cc = cands4(F)
        sig = {}
        for nm in SEL:
            S = smooth(unit(cc[nm].reindex(columns=X.columns)), HL)
            sig[nm] = S
            sig[nm + "|orth"] = orth(S, sl)
        sig["COMP_raw"] = sum(unit(sig[k]) for k in COMP)
        sig["COMP_orth"] = sum(unit(sig[k + "|orth"]) for k in COMP)
        sig["COMP_orth_x"] = orth(sig["COMP_raw"], sl)
        idxBC = F.index[n - 1600:]
        idxA = F.index[n - 4200: n - 1600]
        for nm, S in sig.items():
            U = unit(S)
            net = pnl(U, X, 5.0)["net"]
            rec = [sharpe(net.reindex(idxA)), sharpe(net.reindex(idxBC))]
            rec += [float(row_cos(U.loc[idxBC], sl[k].loc[idxBC]).mean()) for k in ("S06", "S07", "S08")]
            out.setdefault(nm, []).append(rec)
        for i, a in enumerate(COMP):
            for b in COMP[i + 1:]:
                geo.setdefault((a, b), []).append(float(row_cos(sig[a].loc[idxBC], sig[b].loc[idxBC]).mean()))
        print(p, flush=True)
    rows = []
    for nm, v in out.items():
        a = np.array(v, float)
        rows.append(dict(name=nm, SRA=np.nanmean(a[:, 0]), posA=np.nanmean(a[:, 0] > 0), SRBC=np.nanmean(a[:, 1]),
                         tBC=np.nanmean(a[:, 1]) / np.nanstd(a[:, 1]) * np.sqrt(len(a)), posBC=np.nanmean(a[:, 1] > 0),
                         c06=np.nanmean(a[:, 2]), c07=np.nanmean(a[:, 3]), c08=np.nanmean(a[:, 4])))
    df = pd.DataFrame(rows).sort_values("SRBC", ascending=False)
    df.to_csv("research/agents/F5F6_NONLINEAR_screen5.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(4).to_string())
    for k, v in geo.items():
        print("cos", k, round(float(np.mean(v)), 3))


if __name__ == "__main__" and sys.argv[1] == "screen5":
    screen5()


# ----------------------------------------------------------------------------- stage B (evaluate)
def fn_comp4(F):
    """Slow-characteristic composite: |funding| level, funding vol, F5 skew, co-skewness (equal-weight unit sum)."""
    g = blocks(F)
    P = (g[1] + g[2] + g[3] + g[4]) / 4
    dP = P.diff(1)
    m = dP.mean(axis=1)
    M = pd.DataFrame(np.repeat(m.to_numpy()[:, None], dP.shape[1], 1), index=dP.index, columns=dP.columns)
    f5, f6 = g[5], g[6]
    a = rank(f6.abs().rolling(336, min_periods=168).mean())
    b = rank(f6.diff().rolling(336, min_periods=168).std())
    c = rank(f5.rolling(672, min_periods=336).skew())
    Mc = M - M.rolling(672, min_periods=336).mean()
    dc = dP - dP.rolling(672, min_periods=336).mean()
    d = rank((dc * Mc * Mc).rolling(672, min_periods=336).mean()
             / (dP.rolling(672, min_periods=336).std() * M.rolling(672, min_periods=336).std() ** 2))
    return unit(a) + unit(b) + unit(c) + unit(d)


EVALS = {"comp4": (fn_comp4, 672.0)}


def run_eval(names):
    for nm in names:
        fn, hl = EVALS[nm]
        m = evaluate(fn, f"F5F6NL2_{nm}", hl=hl, warm=int(4 * hl + 1400), notes="agent F5F6_NONLINEAR round 2")
        m["admissible"] = admissible(m)
        with open("research/agents/F5F6_NONLINEAR_evals.jsonl", "a") as fh:
            fh.write(json.dumps({k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in m.items()}) + "\n")
        print(nm, "admissible", m["admissible"], flush=True)


if __name__ == "__main__" and sys.argv[1] == "eval":
    run_eval(sys.argv[2:])


# ----------------------------------------------------------------------------- screen 6: recency-trend extrapolation
# eval 3 (comp4) showed pre-scoring segments mis-predict: comp4 was + in A/B/C but -0.0034 on scored rows.
# r07: IC drifts monotonically with row age. Screen 6 measures SR in 6 age bins (scored block dropped first)
# and extrapolates a weighted linear trend in age to the scored block centre. Calibrated on the 6 known evals.
BINS = [(24200, 19200), (19200, 14200), (14200, 9200), (9200, 4200), (4200, 1600), (1600, 0)]  # truncated coords


def screen6():
    out = {}
    xc = np.array([-(a + b) / 2 for a, b in BINS])
    wts = np.array([a - b for a, b in BINS], float)
    for p in dev_periods():
        D = _load(p)
        n = len(D["features"])
        D = {k: v.iloc[: n - 800] for k, v in D.items()}
        F, X = D["features"], D["returns"]
        n = len(F)
        cc = {**legs_raw(F), **cands4(F), "comp4": fn_comp4(F)}
        for name, C in cc.items():
            hl = LEG_HL[name[4:]] if name.startswith("LEG") else HL
            net = pnl(smooth(unit(C.reindex(columns=X.columns)), hl), X, 5.0)["net"].to_numpy()
            # net is aligned to F.index[2:]
            srs = []
            for a, b in BINS:
                lo, hi = n - a - 2, n - b - 2
                srs.append(sharpe(net[max(lo, 0): hi]))
            srs = np.array(srs)
            b1, b0 = np.polyfit(xc, srs, 1, w=np.sqrt(wts))
            out.setdefault(name, []).append(list(srs) + [b0 + b1 * 400.0])
        print(p, flush=True)
    rows = []
    for name, v in out.items():
        a = np.array(v, float)
        e = a[:, -1]
        r = dict(name=name)
        for j in range(len(BINS)):
            r[f"b{j}"] = np.nanmean(a[:, j])
        r["extrap"], r["ext_t"], r["ext_pos"] = e.mean(), e.mean() / e.std() * np.sqrt(len(e)), (e > 0).mean()
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("extrap", ascending=False)
    df.to_csv("research/agents/F5F6_NONLINEAR_screen6.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(4).to_string())


if __name__ == "__main__" and sys.argv[1] == "screen6":
    screen6()
