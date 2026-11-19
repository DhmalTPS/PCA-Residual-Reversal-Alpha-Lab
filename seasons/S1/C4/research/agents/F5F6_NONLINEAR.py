"""Family F5F6_NONLINEAR - agent script.

PLAN (written before any evaluate() call)
  Stage A (screen, NO evaluate() calls, scored rows never touched): every construction below is computed on each
    dev file, but scored only on that file's PRE-SCORING recent segment (rows -5000..-900 from the end), which
    r07 shows carries the same sign as the scored rows. Metrics: net 5bp Sharpe of the hl-672 smoothed unit
    score, IC vs target, cosine with the smoothed occupied legs, and the same for the version residualised
    per hour on the 3 raw occupied legs. ~40 pre-specified constructions (count reported).
  Stage B (evaluate(), HARD LIMIT 12, planned as):
    1-4  : top-4 screen constructions by residual-version screen Sharpe that have max|cos|<=0.35 (raw or resid).
    5-8  : reserved for the best 2 with sign/horizon robustness (hl 336) and 2 sensible combos of survivors.
    9-12 : reserve (not to be used unless a stage-B result is borderline and a pre-stated variant is warranted).
  Report the top 3 by evaluate() honestly, admissible or not.
"""
from __future__ import annotations

import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from lab.honest import rank, zx, unit, smooth, residualize, blocks, occupied_raw, dev_periods, _load, evaluate  # noqa
from lab.backtest import pnl, sharpe
from lab.metrics import ic_series, row_cos

HL = 672.0


def roll_mean(df, w):
    return df.rolling(w, min_periods=max(2, w // 2)).mean()


def roll_std(df, w):
    return df.rolling(w, min_periods=max(2, w // 2)).std()


def cands(F):
    g = blocks(F)
    r = {i: rank(g[i]) for i in g}
    P = (g[1] + g[2] + g[3] + g[4]) / 4
    rP = (r[1] + r[2] + r[3] + r[4]) / 4
    f5, f6 = g[5], g[6]
    c = {}
    # --- F6 dynamics
    c["f6_lvl"] = r[6]
    c["f6_d24"] = rank(f6.diff(24))
    c["f6_d120"] = rank(f6.diff(120))
    c["f6_surpr168"] = rank(f6 - roll_mean(f6, 168))
    c["f6_ma168"] = rank(roll_mean(f6, 168))
    c["f6_ma24"] = rank(roll_mean(f6, 24))
    c["f6_z_ts"] = rank((f6 - roll_mean(f6, 336)) / roll_std(f6, 336))
    c["f6_vol168"] = rank(roll_std(f6.diff(), 168))
    c["f6_sign_lvl"] = rank(np.sign(f6) * roll_mean(np.sign(f6), 72))
    # --- F5 dynamics
    c["f5_ma24"] = rank(roll_mean(f5, 24))
    c["f5_ma168"] = rank(roll_mean(f5, 168))
    c["f5_vol168"] = rank(roll_std(f5, 168))
    c["f5_tail"] = r[5].where(r[5].abs() > 0.6, 0.0)
    c["f5_signed_sqrt"] = rank(np.sign(f5) * np.sqrt(f5.abs()))
    c["f5_clip_ma24"] = rank(roll_mean(f5.clip(-2, 2), 24))
    c["f5_d1"] = rank(f5.diff(1))
    c["f5_absma24"] = rank(roll_mean(f5.abs(), 24))
    # --- interactions with F1-4
    c["rev_x_f5hi"] = -rP * (r[5] > 0.3)
    c["rev_x_f5lo"] = -rP * (r[5] < -0.3)
    c["rP_times_r5"] = rP * r[5]
    c["d24_x_absf5"] = rank(np.sign(P.diff(24)) * rank(f5.abs()).add(1))
    c["d24_x_f5vol"] = rank(P.diff(24) * rank(roll_std(f5, 168)).add(1))
    c["rP_times_r6"] = rP * r[6]
    c["d120_times_r6"] = rank(P.diff(120)) * r[6]
    c["f6_minus_rP"] = r[6] - rP  # carry vs momentum spread
    c["f6_x_d24sign"] = r[6] * np.sign(P.diff(24))
    c["sign24_mag_hi"] = -np.sign(P.diff(24)) * (rank(P.diff(24).abs()) > 0.5)
    c["sign24_mag_lo"] = -np.sign(P.diff(24)) * (rank(P.diff(24).abs()) < -0.5)
    c["rP_tails"] = -rP.where(rP.abs() > 0.6, 0.0)
    c["rP_mid"] = -rP.where(rP.abs() <= 0.6, 0.0)
    c["cond_rev_f6"] = -rank(P.diff(24)) * (r[6] > 0)
    # --- odd/even spreads inside F1-4
    c["f1_m_f2"] = rank(g[1] - g[2])
    c["f3_m_f4"] = rank(g[3] - g[4])
    c["f13_m_f24"] = rank(g[1] + g[3] - g[2] - g[4])
    c["f1_m_f3"] = rank(g[1] - g[3])
    c["f2_m_f4"] = rank(g[2] - g[4])
    c["f1_m_f4"] = rank(g[1] - g[4])
    c["f13m24_ma24"] = rank(roll_mean(g[1] + g[3] - g[2] - g[4], 24))
    c["disp_F14"] = rank(pd.concat([g[i] for i in (1, 2, 3, 4)]).groupby(level=0).std())
    return c


def screen():
    out = {}
    for p in dev_periods():
        D = _load(p, 5000 + 3400)
        F, X, T = D["features"], D["returns"], D["target"]
        seg = F.index[len(F) - 5000: len(F) - 900]
        legs = occupied_raw(F)
        legs = {k: v.reindex(columns=X.columns) for k, v in legs.items()}
        sl = {k: smooth(v, HL).loc[seg] for k, v in legs.items()}
        for name, C in cands(F).items():
            C = C.reindex(columns=X.columns)
            for ver, CC in (("raw", unit(C)), ("res", unit(residualize(C, list(legs.values()))))):
                S = smooth(CC, HL)
                Ss = S.loc[seg]
                sr = sharpe(pnl(Ss, X.loc[seg], 5.0)["net"])
                ic = float(ic_series(CC, T, 1).reindex(seg).mean())
                cs = [float(row_cos(Ss, sl[k]).mean()) for k in ("S06", "S07", "S08")]
                out.setdefault((name, ver), []).append([sr, ic] + cs)
        print(p, flush=True)
    rows = []
    for (name, ver), v in out.items():
        a = np.array(v)
        rows.append(dict(name=name, ver=ver, SR=a[:, 0].mean(), t=a[:, 0].mean() / a[:, 0].std() * np.sqrt(len(a)),
                         pos=(a[:, 0] > 0).mean(), IC=a[:, 1].mean(), c06=a[:, 2].mean(), c07=a[:, 3].mean(),
                         c08=a[:, 4].mean()))
    df = pd.DataFrame(rows).sort_values("SR", ascending=False)
    df.to_csv("research/agents/F5F6_NONLINEAR_screen.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "screen":
        screen()


def cands2(F):
    """Second, focused screen (pre-scoring segment only): variants around the two stage-A survivors."""
    g = blocks(F)
    r = {i: rank(g[i]) for i in g}
    P = (g[1] + g[2] + g[3] + g[4]) / 4
    f6 = g[6]
    d6 = f6.diff()
    c = {}
    for w in (72, 168, 336, 672):
        c[f"f6_vol{w}"] = rank(roll_std(d6, w))
    c["f6_absd_ma336"] = rank(roll_mean(d6.abs(), 336))
    c["f6_lvlstd336"] = rank(roll_std(f6, 336))
    c["f6_chgfreq336"] = rank(roll_mean((d6 != 0).astype(float), 336))
    c["f6_absmean336"] = rank(roll_mean(f6.abs(), 336))
    c["f6_volratio"] = rank(roll_std(d6, 336) / roll_std(f6, 336))
    c["f5_vol336"] = rank(roll_std(g[5], 336))
    c["f6_x_d120sign"] = r[6] * np.sign(P.diff(120))
    c["f6_x_d24sign"] = r[6] * np.sign(P.diff(24))
    c["combo_vol_x"] = unit(rank(roll_std(d6, 336))) + unit(r[6] * np.sign(P.diff(24)))
    return c


def screen2():
    out = {}
    for p in dev_periods():
        D = _load(p, 5000 + 3400)
        F, X, T = D["features"], D["returns"], D["target"]
        seg = F.index[len(F) - 5000: len(F) - 900]
        legs = {k: v.reindex(columns=X.columns) for k, v in occupied_raw(F).items()}
        sl = {k: smooth(v, HL).loc[seg] for k, v in legs.items()}
        for name, C in cands2(F).items():
            C = C.reindex(columns=X.columns)
            for ver, CC in (("raw", unit(C)), ("res", unit(residualize(C, list(legs.values()))))):
                Ss = smooth(CC, HL).loc[seg]
                sr = sharpe(pnl(Ss, X.loc[seg], 5.0)["net"])
                ic = float(ic_series(CC, T, 1).reindex(seg).mean())
                cs = [float(row_cos(Ss, sl[k]).mean()) for k in ("S06", "S07", "S08")]
                out.setdefault((name, ver), []).append([sr, ic] + cs)
    rows = []
    for (name, ver), v in out.items():
        a = np.array(v)
        rows.append(dict(name=name, ver=ver, SR=a[:, 0].mean(), t=a[:, 0].mean() / a[:, 0].std() * np.sqrt(len(a)),
                         pos=(a[:, 0] > 0).mean(), IC=a[:, 1].mean(), c06=a[:, 2].mean(), c07=a[:, 3].mean(),
                         c08=a[:, 4].mean()))
    df = pd.DataFrame(rows).sort_values("SR", ascending=False)
    df.to_csv("research/agents/F5F6_NONLINEAR_screen2.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string())


if __name__ == "__main__" and sys.argv[1] == "screen2":
    screen2()


# ----------------------------------------------------------------------------- stage B candidates (evaluate())
def fn_f6_absmean336(F):
    f6 = blocks(F)[6]
    return rank(f6.abs().rolling(336, min_periods=168).mean())


def fn_f6_lvlstd336(F):
    f6 = blocks(F)[6]
    return rank(f6.rolling(336, min_periods=168).std())


EVALS = {
    "f6_absmean336": fn_f6_absmean336,
    "f6_lvlstd336": fn_f6_lvlstd336,
}

if __name__ == "__main__" and sys.argv[1] == "eval":
    from lab.honest import admissible
    for nm in sys.argv[2:]:
        m = evaluate(EVALS[nm], f"F5F6NL_{nm}", notes="agent F5F6_NONLINEAR")
        m["admissible"] = admissible(m)
        with open("research/agents/F5F6_NONLINEAR_evals.jsonl", "a") as fh:
            fh.write(json.dumps({k: (v if not isinstance(v, (np.floating,)) else float(v)) for k, v in m.items()}) + "\n")
        print(nm, "admissible", m["admissible"], flush=True)


# ----------------------------------------------------------------------------- stage A3: recency-drift extrapolation
# Evals 1-2 showed the pre-scoring segment mis-predicts F6-scale signals (sign flip). r07: per-file IC drifts
# monotonically with row age. Screen 3 measures the drift on 5 in-file segments (never the scored rows) and
# extrapolates a linear trend in row age to the scored block (centre -400). Calibrated on the occupied legs.
SEGS = [(25000, 20000), (20000, 15000), (15000, 10000), (10000, 5000), (5000, 900)]


def screen3():
    out = {}
    xc = np.array([-(a + b) / 2 for a, b in SEGS])
    for p in dev_periods():
        D = _load(p)
        F, X, T = D["features"], D["returns"], D["target"]
        n = len(F)
        legs = {k: v.reindex(columns=X.columns) for k, v in occupied_raw(F).items()}
        cc = {**{f"LEG_{k}": v for k, v in legs.items()}, **cands(F), **cands2(F)}
        for name, C in cc.items():
            C = C.reindex(columns=X.columns)
            vers = [("raw", unit(C))]
            if not name.startswith("LEG"):
                vers.append(("res", unit(residualize(C, list(legs.values())))))
            for ver, CC in vers:
                S = smooth(CC, HL)
                net = pnl(S, X, 5.0)["net"]
                srs = [sharpe(net.iloc[n - a: n - b]) for a, b in SEGS]
                b1, b0 = np.polyfit(xc, srs, 1)
                out.setdefault((name, ver), []).append(srs + [b0 + b1 * (-400.0)])
        print(p, flush=True)
    rows = []
    for (name, ver), v in out.items():
        a = np.array(v)
        e = a[:, -1]
        rows.append(dict(name=name, ver=ver, s1=a[:, 0].mean(), s2=a[:, 1].mean(), s3=a[:, 2].mean(),
                         s4=a[:, 3].mean(), s5=a[:, 4].mean(), extrap=e.mean(),
                         ext_t=e.mean() / e.std() * np.sqrt(len(e)), ext_pos=(e > 0).mean()))
    df = pd.DataFrame(rows).sort_values("extrap", ascending=False)
    df.to_csv("research/agents/F5F6_NONLINEAR_screen3.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.round(4).to_string())


if __name__ == "__main__" and sys.argv[1] == "screen3":
    screen3()
