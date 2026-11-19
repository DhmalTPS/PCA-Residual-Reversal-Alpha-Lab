"""Family F5F6_NONLINEAR - round 3 (resumed after interruption, 2026-10-08).

BUDGET: 3 evaluate() calls already spent (f6_absmean336 +0.0068, f6_lvlstd336 -0.0086, comp4 -0.0034).
HARD LIMIT 12 -> 9 left for this round. None of the 3 is repeated.

LESSON: pre-scoring segment screens (screen2/4/5/6) did NOT predict honest scores (comp4 screen +0.0096 -> honest
-0.0034; lvlstd screen +0.009 -> honest -0.0086). In-file history is a hindsight reconstruction. => no further
screening; this round evaluates SIMPLE, economically-motivated constructions, pre-declared below, signs fixed a priori.

Occupied directions: odd F1-4 (reversal, S06/S08), even F1-4 (|z| extremeness, S07), F5 level (S08), F6 level (S06,
weight 0.5). Unoccupied, tested here: PURE INTERACTIONS (odd return x even F5/F6 state) and an orthogonalised F6
characteristic. R := -rank(P.diff(120)) with P = mean(F1..F4) (5-day reversal, sign +).

PLAN (pre-declared, in order; all hl 672 unless stated):
  E1 crev_f6ext   R * rank(|zx(F6)|)              reversal stronger where funding is extreme (crowded leverage);
                                                 |zx| is invariant to the documented sign flip of the F6 level.
  E2 crev_f5act   R * rank(mean_120h F5)          Campbell-Grossman-Wang: moves on abnormal volume (liquidity /
                                                 liquidation-driven) revert more.
  E3 crev_f6sgn   R * rank(F6)                    moves with high funding (levered longs paying) revert; low funding
                                                 -> continuation. Odd x odd = even function of the joint state.
  E4 f6abs_orth   smooth_672(unit(rank(mean_336 |F6|))) residualised per hour on the three smoothed occupied legs,
                  evaluate(hl=0). Tests whether the only positive honest F6 result (+0.0068, cos06 0.47) survives
                  orthogonalisation.
  E5 crev24_f5spk -rank(P.diff(24)) * rank(max_24h F5)  short-horizon: 1-day moves accompanied by a volume spike
                                                 revert (liquidation cascades).
  Conditional (decided only from E1-E5 results, rules fixed now):
  E6  best of E1-E5 by SR5 (if SR5 > 0): same construction at hl 336 (horizon robustness).
  E7  best of E1-E5 by SR5 (if SR5 > 0 and max|cos| > 0.25): orth version (smoothed in fn, residualised on smoothed
      legs, hl=0).
  E8  equal-weight unit sum of the E1-E5 constructions with SR5 > 0.004 (only if >= 2 such), hl 672.
  E9  reserve - used only if a result is borderline-admissible; otherwise left unspent.
  Report the top 3 honestly, admissible or not.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from lab.honest import (rank, zx, unit, smooth, residualize, blocks, occupied_raw, evaluate, admissible,  # noqa
                        LEG_HL)

LOG = "research/agents/F5F6_NONLINEAR_evals.jsonl"


def _P(g):
    return (g[1] + g[2] + g[3] + g[4]) / 4


def fn_crev_f6ext(F):
    g = blocks(F)
    return -rank(_P(g).diff(120)) * rank(zx(g[6]).abs())


def fn_crev_f5act(F):
    g = blocks(F)
    return -rank(_P(g).diff(120)) * rank(g[5].rolling(120, min_periods=60).mean())


def fn_crev_f6sgn(F):
    g = blocks(F)
    return -rank(_P(g).diff(120)) * rank(g[6])


def fn_crev24_f5spk(F):
    g = blocks(F)
    return -rank(_P(g).diff(24)) * rank(g[5].rolling(24, min_periods=12).max())


def orth_smoothed(fn, hl):
    """Smooth the unit score inside fn and residualise per hour on the three smoothed occupied legs (use hl=0)."""
    def f(F):
        S = smooth(unit(fn(F)), hl)
        legs = {k: smooth(v.reindex(columns=S.columns), LEG_HL[k]) for k, v in occupied_raw(F).items()}
        return residualize(S, [legs[k] for k in ("S06", "S07", "S08")])
    return f


def fn_f6abs(F):
    g = blocks(F)
    return rank(g[6].abs().rolling(336, min_periods=168).mean())


EVALS = {
    "crev_f6ext": (fn_crev_f6ext, 672.0, None),
    "crev_f5act": (fn_crev_f5act, 672.0, None),
    "crev_f6sgn": (fn_crev_f6sgn, 672.0, None),
    "f6abs_orth": (orth_smoothed(fn_f6abs, 672.0), 0.0, 4 * 672 + 1400),
    "crev24_f5spk": (fn_crev24_f5spk, 672.0, None),
}


def run_eval(names):
    for nm in names:
        fn, hl, warm = EVALS[nm]
        warm = warm if warm is not None else int(4 * hl + 1400)
        m = evaluate(fn, f"F5F6NL3_{nm}", hl=hl, warm=warm, notes="agent F5F6_NONLINEAR round 3")
        m["admissible"] = admissible(m)
        with open(LOG, "a") as fh:
            fh.write(json.dumps({k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in m.items()}) + "\n")
        print(nm, "admissible", m["admissible"], flush=True)


if __name__ == "__main__" and sys.argv[1] == "eval":
    run_eval(sys.argv[2:])


# ----------------------------------------------------------------------------- conditional stage (after E1-E5)
# E1 crev_f6ext +0.0059, E2 crev_f5act +0.0111 (ci_lo +0.0016, max|cos| 0.22), E3 -0.0079, E4 -0.0072, E5 -0.0034.
# Rules applied: E6 = crev_f5act at hl 336; E7 skipped (max|cos| 0.22 <= 0.25); E8 = unit(E1)+unit(E2) (both > 0.004).
def fn_combo_e1e2(F):
    return unit(fn_crev_f6ext(F)) + unit(fn_crev_f5act(F))


EVALS["crev_f5act_hl336"] = (fn_crev_f5act, 336.0, None)
EVALS["combo_e1e2"] = (fn_combo_e1e2, 672.0, None)

if __name__ == "__main__" and sys.argv[1] == "eval2":
    run_eval(sys.argv[2:])

# ----------------------------------------------------------------------------- RESULT (round 3, final)
# E6 crev_f5act hl336 +0.0076 (hl sensitivity: 672 -> 336 loses 1/3). E8 combo_e1e2 +0.0141 [+0.0050,+0.0234],
# SR10 +0.0103, but periods+ 0.55, DSR 0.24, IC_T t 1.0 -> NOT admissible. Half-sample (diagnostic, no new trial,
# _f5f6_v3_halves.py): combo H1 +0.0027 / H2 +0.0216; crev_f5act H1 -0.0029 / H2 +0.0204 -> edge concentrated in
# later periods, fails the skeptic half-sample test. E7 skipped by rule, E9 left unspent (nothing borderline).
# Family total: 10 of 12 evaluate() calls used. Verdict: NULL - no admissible F5/F6 nonlinear construction.
