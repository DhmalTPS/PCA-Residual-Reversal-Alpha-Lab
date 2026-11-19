"""Family TIMESCALE - short/medium-horizon reversal & continuation that is NOT the 120h reversal.

PLAN (pre-registered, written before any evaluate() call; HARD LIMIT 12 evaluate() calls):
  Stage 0 (outcome-free, NOT a trial): geometry pre-screen. For each construction x half-life compute the
      mean per-hour cosine of the smoothed candidate with the cached smoothed occupied legs (S06/S07/S08)
      and the turnover of the smoothed position on 8 dev periods. Uses features only (no returns, no
      target) -> no data snooping of performance. Keep only (construction, hl) with max|cos| <= 0.35.
  Stage 1 (evaluate calls 1..8): the orthogonal survivors, priority order
      E1  d24 reversal of mean F1-4, residualised per hour on d120 + occupied raw legs
      E2  d24 F3 reversal raw (honest-screen lead, IC -0.015, 83% stable)  [geometry permitting]
      E3  term-structure contrast  -(rank d24 - rank d120)  residualised on occupied legs
      E4  d6 reversal residualised on d24/d120/occupied
      E5  d48 reversal residualised on d120/occupied
      E6  acceleration -(d24 - lagged d24) (second difference) residualised
      E7  F5 / F6 short-horizon change (d24) residualised on occupied legs
      E8  best of E1..E7 sign-agnostic? NO - signs fixed a priori (reversal = negative of change); a
          negative SR is reported as such, never flipped post hoc.
  Stage 2 (calls 9..12, reserved): half-life refinement of the single best Stage-1 survivor
      (one shorter, one longer hl) and one blend of the two best orthogonal survivors. Stop early
      if Stage 1 shows nothing with SR5 > 0.008 (null result).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from lab import honest as h  # noqa: E402
from lab.honest import rank, zx, unit, smooth, residualize, blocks, occupied_raw  # noqa: E402
from lab.metrics import row_cos  # noqa: E402


# ----------------------------------------------------------------------------- constructions
def _G(F):
    g = blocks(F)
    return g, (g[1] + g[2] + g[3] + g[4]) / 4.0


def _occ(F):
    o = occupied_raw(F)
    return [o["S06"], o["S07"], o["S08"]]


def c_d24_res(F):
    g, G = _G(F)
    return -residualize(rank(G.diff(24)), [rank(G.diff(120))] + _occ(F))


def c_d24f3(F):
    g = blocks(F)
    return -rank(g[3].diff(24))


def c_d24f3_res(F):
    g = blocks(F)
    return -residualize(rank(g[3].diff(24)), [rank(g[3].diff(120))] + _occ(F))


def c_term(F):
    g, G = _G(F)
    return -residualize(rank(G.diff(24)) - rank(G.diff(120)), _occ(F))


def c_d6_res(F):
    g, G = _G(F)
    return -residualize(rank(G.diff(6)), [rank(G.diff(24)), rank(G.diff(120))] + _occ(F))


def c_d48_res(F):
    g, G = _G(F)
    return -residualize(rank(G.diff(48)), [rank(G.diff(120))] + _occ(F))


def c_acc(F):
    g, G = _G(F)
    d = G.diff(24)
    return -residualize(rank(d - d.shift(24)), _occ(F))


def c_f5d24(F):
    g = blocks(F)
    return -residualize(rank(g[5].diff(24)), _occ(F))


def c_f6d24(F):
    g = blocks(F)
    return -residualize(rank(g[6].diff(24)), _occ(F))


def c_d24_raw(F):
    g, G = _G(F)
    return -rank(G.diff(24))


def c_d6_raw(F):
    g, G = _G(F)
    return -rank(G.diff(6))


CONS = {k[2:]: v for k, v in dict(globals()).items() if k.startswith("c_") and callable(v)}


# ----------------------------------------------------------------------------- stage 0 geometry
def geometry(periods=("003", "007", "011", "015", "019", "023", "026", "029"), hls=(48, 96, 168, 336, 672)):
    rows = []
    for p in periods:
        D = h._load(p, h.VAL + 4 * 672 + 600)
        F = D["features"]
        cols = D["returns"].columns
        vi = F.index[len(F) - h.VAL:]
        legs = h.occupied_smoothed(p, 4 * 672 + 600)
        for name, fn in CONS.items():
            U = unit(fn(F).reindex(index=F.index, columns=cols))
            for hl in hls:
                Sm = smooth(U, hl).loc[vi]
                r = {"p": p, "c": name, "hl": hl}
                for k in ("S06", "S07", "S08"):
                    r[k] = float(row_cos(Sm, legs[k].reindex(index=Sm.index, columns=Sm.columns)).mean())
                r["turn"] = float((Sm.diff().abs().sum(axis=1)).mean())
                r["gross"] = float(Sm.abs().sum(axis=1).mean())
                rows.append(r)
        print("done", p, flush=True)
    d = pd.DataFrame(rows).groupby(["c", "hl"]).mean(numeric_only=True)
    d["maxcos"] = d[["S06", "S07", "S08"]].abs().max(axis=1)
    d["turn_per_gross"] = d["turn"] / d["gross"]
    pd.set_option("display.width", 200)
    print(d.round(3).to_string())
    d.to_csv("research/agents/timescale_geometry.csv")


def run(name, fn, hl, notes=""):
    m = h.evaluate(fn, f"timescale_{name}_hl{hl}", hl=hl, notes=notes)
    m["admissible"] = h.admissible(m)
    with open("research/agents/timescale_results.jsonl", "a") as f:
        f.write(json.dumps({k: (v if not isinstance(v, float) else round(v, 6)) for k, v in m.items()}) + "\n")
    return m


if __name__ == "__main__":
    if sys.argv[1] == "geom":
        geometry()
    elif sys.argv[1] == "eval":
        for spec in sys.argv[2:]:
            c, hl = spec.split("@")
            run(c, CONS[c], float(hl))


# ----------------------------------------------------------------------------- post-smoothing residualisation
# Stage-0 finding: residualising the RAW score per hour on raw legs leaves a residual that, once smoothed,
# is -0.86 cosine with S06 (the projection removes the fast part only). The scored object is the SMOOTHED
# position, so orthogonalise there: smooth the candidate causally inside fn, project out the causally
# smoothed occupied legs (same half-lives as the cached legs), and evaluate with hl=0 (no extra smoothing).
def post(raw_fn, hl, extra=()):
    def fn(F):
        cols = list(dict.fromkeys(F.columns.get_level_values(-1)))
        U = unit(raw_fn(F).reindex(columns=cols))
        Sm = smooth(U, hl)
        o = occupied_raw(F)
        legs = [smooth(o[k].reindex(columns=cols), h.LEG_HL[k]) for k in ("S06", "S07", "S08")]
        legs += [smooth(unit(e(F).reindex(columns=cols)), hl) for e in extra]
        return residualize(Sm, legs)
    return fn


def r_d24(F):
    g, G = _G(F)
    return -rank(G.diff(24))


def r_d6(F):
    g, G = _G(F)
    return -rank(G.diff(6))


def r_d48(F):
    g, G = _G(F)
    return -rank(G.diff(48))


def r_term(F):
    g, G = _G(F)
    return -(rank(G.diff(24)) - rank(G.diff(120)))


def r_f5d24(F):
    return -rank(blocks(F)[5].diff(24))


def r_f6d24(F):
    return -rank(blocks(F)[6].diff(24))


RAW = {"d24": r_d24, "d6": r_d6, "d48": r_d48, "term": r_term, "f5d24": r_f5d24, "f6d24": r_f6d24}
POST = {f"P{k}_{hl}": post(v, hl) for k, v in RAW.items() for hl in (48, 168, 336)}


def geometry_post(periods=("003", "011", "019", "026", "029")):
    rows = []
    for p in periods:
        D = h._load(p, h.VAL + 4 * 672 + 600)
        F = D["features"]
        cols = D["returns"].columns
        vi = F.index[len(F) - h.VAL:]
        legs = h.occupied_smoothed(p, 4 * 672 + 600)
        for name, fn in POST.items():
            Sm = smooth(unit(fn(F).reindex(index=F.index, columns=cols)), 0).loc[vi]
            r = {"p": p, "c": name}
            for k in ("S06", "S07", "S08"):
                r[k] = float(row_cos(Sm, legs[k].reindex(index=Sm.index, columns=Sm.columns)).mean())
            r["turn_per_gross"] = float(Sm.diff().abs().sum(axis=1).mean() / Sm.abs().sum(axis=1).mean())
            # how much of the smoothed candidate survives the projection
            U = unit(RAW[name[1:].rsplit("_", 1)[0]](F).reindex(columns=cols))
            S0 = smooth(U, float(name.rsplit("_", 1)[1])).loc[vi]
            r["keep"] = float(row_cos(Sm, S0).mean())
            rows.append(r)
        print("done", p, flush=True)
    d = pd.DataFrame(rows).groupby("c").mean(numeric_only=True)
    print(d.round(3).to_string())


if __name__ == "__main__" and sys.argv[1] == "geompost":
    geometry_post()


# REVISED STAGE-1 PLAN (after outcome-free geometry; still <= 12 evaluate calls total):
#   1 Pd24_168  2 d24_raw@168  3 Pd6_48  4 Pd48_336  5 Pterm_168  6 Pf6d24_336  7 Pf5d24_168  8 Pd24_48
#   9-12 reserved (hl refinement / blend of best two). P* run with evaluate hl=0 (smoothing is inside fn).
WARM_POST = 4 * 672 + 700


def run_post(key, notes="post-smoothing residual on smoothed S06/S07/S08"):
    m = h.evaluate(POST[key], f"timescale_{key}", hl=0.0, warm=WARM_POST, notes=notes)
    m["admissible"] = h.admissible(m)
    with open("research/agents/timescale_results.jsonl", "a") as f:
        f.write(json.dumps({k: (v if not isinstance(v, float) else round(v, 6)) for k, v in m.items()}) + "\n")
    return m


if __name__ == "__main__" and sys.argv[1] == "evalpost":
    for k in sys.argv[2:]:
        run_post(k)


# ----------------------------------------------------------------------------- L1-cost execution (no-trade band)
# Calls 1-4 show the d24 reversal has gross Sharpe ~ +0.013 but cost ~ 1x gross at any EW half-life
# (EW smoothing shrinks gross together with turnover). Under LINEAR costs the optimal tracker of an aim
# portfolio is a no-trade band (not an EW filter): p_t = clip(p_{t-1}, A_t - b, A_t + b). Causal recursion.
def band_path(A: np.ndarray, b: float) -> np.ndarray:
    P = np.empty_like(A)
    p = A[0].copy()
    for t in range(len(A)):
        a = A[t]
        p = np.minimum(np.maximum(p, a - b), a + b)
        P[t] = p
    return P


def banded(raw_fn, hl, b):
    def fn(F):
        cols = list(dict.fromkeys(F.columns.get_level_values(-1)))
        Sm = unit(smooth(unit(raw_fn(F).reindex(columns=cols)), hl))  # unit aim each hour
        P = band_path(Sm.to_numpy(float), b)
        return pd.DataFrame(P, index=Sm.index, columns=cols)
    return fn


def geometry_band(periods=("003", "011", "019", "026", "029")):
    rows = []
    for p in periods:
        D = h._load(p, h.VAL + 4 * 672 + 600)
        F = D["features"]
        cols = D["returns"].columns
        vi = F.index[len(F) - h.VAL:]
        legs = h.occupied_smoothed(p, 4 * 672 + 600)
        for hl in (24, 48, 168):
            for b in (0.0, 0.05, 0.1, 0.15, 0.2):
                Sm = smooth(unit(banded(r_d24, hl, b)(F).reindex(index=F.index, columns=cols)), 0)
                A = unit(smooth(unit(r_d24(F).reindex(columns=cols)), hl))
                Sm, A = Sm.loc[vi], A.loc[vi]
                r = {"p": p, "hl": hl, "b": b,
                     "turn_per_gross": float(Sm.diff().abs().sum(axis=1).mean() / Sm.abs().sum(axis=1).mean()),
                     "cos_aim": float(row_cos(Sm, A).mean()),
                     "cos_aim_lag24": float(row_cos(Sm, A.shift(-24)).mean())}  # diagnostic only, not a signal
                for k in ("S06", "S07", "S08"):
                    r[k] = float(row_cos(Sm, legs[k].reindex(index=Sm.index, columns=Sm.columns)).mean())
                rows.append(r)
        print("done", p, flush=True)
    d = pd.DataFrame(rows).groupby(["hl", "b"]).mean(numeric_only=True)
    print(d.round(3).to_string())


if __name__ == "__main__" and sys.argv[1] == "geomband":
    geometry_band()


# ----------------------------------------------------------------------------- vol-scaled reversal (calls 5-7)
def r_d24v(F):
    g, G = _G(F)
    vol = G.diff(1).rolling(336, min_periods=168).std()
    return -rank(G.diff(24) / vol)


def r_multiv(F):
    g, G = _G(F)
    vol = G.diff(1).rolling(336, min_periods=168).std()
    return -(rank(G.diff(6) / (vol * 6 ** 0.5)) + rank(G.diff(24) / (vol * 24 ** 0.5)) + rank(G.diff(48) / (vol * 48 ** 0.5))) / 3


CONS["d24v"] = r_d24v
CONS["multiv"] = r_multiv
BAND = {"Bd24v_168_10": banded(r_d24v, 168, 0.10)}

if __name__ == "__main__" and sys.argv[1] == "evalband":
    for k in sys.argv[2:]:
        m = h.evaluate(BAND[k], f"timescale_{k}", hl=0.0, warm=WARM_POST, notes="no-trade band execution")
        m["admissible"] = h.admissible(m)
        with open("research/agents/timescale_results.jsonl", "a") as f:
            f.write(json.dumps({kk: (v if not isinstance(v, float) else round(v, 6)) for kk, v in m.items()}) + "\n")

if __name__ == "__main__" and sys.argv[1] == "eval2":
    for spec in sys.argv[2:]:
        c, hl = spec.split("@")
        run(c, CONS[c], float(hl))

BAND["Bd24_168_15"] = banded(r_d24, 168, 0.15)
if __name__ == "__main__" and sys.argv[1] == "evalband2":
    for k in sys.argv[2:]:
        m = h.evaluate(BAND[k], f"timescale_{k}", hl=0.0, warm=WARM_POST, notes="no-trade band execution")
        m["admissible"] = h.admissible(m)
        with open("research/agents/timescale_results.jsonl", "a") as f:
            f.write(json.dumps({kk: (v if not isinstance(v, float) else round(v, 6)) for kk, v in m.items()}) + "\n")
