"""C5 r06b - analysis of the r06 audit outputs (no new signal construction; reads experiments/c5_r06/)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from lab.metrics import sharpe_se_nw
from lab.official import periods_split
from lab.paths import EXPERIMENTS, REPORTS, settings

OUT = EXPERIMENTS / "c5_r06"
SERVER = {"A": 0.512, "AB": 0.638, "L240": 0.632, "Z": 0.608,
          "S06": float(np.cos(np.radians(16.31))), "S07": float(np.cos(np.radians(38.40))),
          "S08": float(np.cos(np.radians(42.34))), "S09": float(np.cos(np.radians(61.25)))}
CANDS = ["A", "AB", "L240", "Z", "X", "SWD", "SWC", "S06", "S07", "S08", "S09"]
LAG = int(settings()["nw_lag_hours"])
devp, vp = periods_split(settings())
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def sr(cand, ps):
    r = np.concatenate([np.load(OUT / f"pnl_{cand}_{p}.npy") for p in ps])
    s, se, _ = sharpe_se_nw(pd.Series(r), LAG)
    return s, se


# ---------------------------------------------------------------- T1 facts
say("## T1 - net SR @5bp per candidate (NW s.e., lag 48), exact server-judged files")
say("| cand | full dev 001-029 | recent dev 022-029 | vault 030-031 | proxy (10) | provenance caveat |")
say("|---|---|---|---|---|---|")
cav = {"A": "selected on proxy (incl. vault) from 16-grid", "AB": "built from A,B selected on proxy (incl. vault)",
       "L240": "chosen by blocker model + proxy rule", "Z": "weights fit on dev 001-029 -> dev in-sample; vault clean",
       "X": "AB + Z: both caveats", "SWD": "pre-declared, first look", "SWC": "CONTROL only",
       "S06": "C4 probe", "S07": "C4 probe", "S08": "C4 probe", "S09": "our pot entry"}
for c in CANDS:
    cells = []
    for ps in (devp, devp[-8:], vp, devp[-8:] + vp):
        s, se = sr(c, ps)
        cells.append(f"{s:+.4f} ({se:.4f})")
    say(f"| {c} | " + " | ".join(cells) + f" | {cav[c]} |")

# ---------------------------------------------------------------- correlations
Z = np.load(OUT / "hourly_cos.npz")
pairs = list(Z.files)
cmean = {k: float(np.nanmean(Z[k])) for k in pairs}
libs = sorted({k.split("|", 1)[1] for k in pairs} - set(CANDS))
say("\n## Probe x probe time-averaged cross-sectional correlation (rho_hat, scored rows of all 31 periods)")
say("| | " + " | ".join(CANDS) + " |")
say("|---|" + "---|" * len(CANDS))
for a in CANDS:
    say(f"| {a} | " + " | ".join(f"{cmean[f'{a}|{b}']:+.2f}" for b in CANDS) + " |")

# ---------------------------------------------------------------- T2 linearity (time averaging of the exact hourly identity)
say("\n## T2 - linear blend prediction vs actual over the library (X and SWD)")
reg = pd.read_parquet(OUT / "regime.parquet")
w = reg["w"].to_numpy()
cabz = Z["AB|Z"]
for cand, wt in (("X", np.full_like(w, 0.5)), ("SWD", w)):
    nrm = np.sqrt(wt ** 2 + (1 - wt) ** 2 + 2 * wt * (1 - wt) * cabz)
    err = []
    for l in libs:
        pred = np.nanmean((wt * Z[f"AB|{l}"] + (1 - wt) * Z[f"Z|{l}"]) / nrm)
        err.append(cmean[f"{cand}|{l}"] - pred)
    err = np.array(err)
    say(f"{cand}: hourly-identity prediction error over {len(libs)} stand-ins: mean {err.mean():+.4f}, max |err| {np.abs(err).max():.4f}")

# ---------------------------------------------------------------- T3 cross-terms
say("\n## T3 - cross-terms: stand-ins near one component, their correlation with the other")
for near, other in (("Z", "AB"), ("AB", "Z")):
    sel = [l for l in libs if abs(cmean[f"{near}|{l}"]) >= 0.40]
    if not sel:
        say(f"no stand-in with |rho({near},l)| >= 0.40")
        continue
    v = np.array([cmean[f"{other}|{l}"] * np.sign(cmean[f"{near}|{l}"]) for l in sel])
    top = sorted(sel, key=lambda l: -abs(cmean[f"{near}|{l}"]))[:8]
    say(f"{len(sel)} stand-ins with |rho({near},l)| >= 0.40: signed rho({other},l) mean {v.mean():+.3f}, "
        f"range [{v.min():+.3f}, {v.max():+.3f}]")
    for l in top:
        say(f"   {l:24s} rho({near}) {cmean[f'{near}|{l}']:+.3f}  rho({other}) {cmean[f'{other}|{l}']:+.3f}  "
            f"rho(X) {cmean[f'X|{l}']:+.3f}  rho(SWD) {cmean[f'SWD|{l}']:+.3f}  rho(SWC) {cmean[f'SWC|{l}']:+.3f}")

# ---------------------------------------------------------------- T4 calibration
say("\n## T4 - does any library statistic rank the 8 server-measured probes? (Spearman, n=8)")
probes = list(SERVER)
stats = {}
for thr in (0.95, 0.90, 0.80):
    stats[f"max|rho| excl >{thr}"] = [max([abs(cmean[f"{p}|{l}"]) for l in libs if abs(cmean[f"{p}|{l}"]) <= thr] or [0]) for p in probes]
for c in (0.4, 0.5, 0.6):
    stats[f"count |rho|>{c}"] = [sum(abs(cmean[f"{p}|{l}"]) > c for l in libs) for p in probes]
srv = [SERVER[p] for p in probes]
say("probe server-rho: " + ", ".join(f"{p} {SERVER[p]:.3f}" for p in probes))
for k, v in stats.items():
    r, pv = spearmanr(srv, v)
    say(f"  {k:22s}: {', '.join(f'{x:.2f}' if isinstance(x, float) else str(x) for x in v)} | Spearman {r:+.2f} (p {pv:.2f})")

# ---------------------------------------------------------------- T5 regime stationarity
say("\n## T5 - is the hourly correlation with stand-ins the same in both dispersion regimes?")
hi = reg["hi"].to_numpy() > 0.5
say(f"share of high-dispersion hours {hi.mean():.2f}; mean SWD weight on AB {w.mean():.2f}")
for comp in ("AB", "Z"):
    sel = sorted(libs, key=lambda l: -abs(cmean[f"{comp}|{l}"]))[:10]
    d = []
    for l in sel:
        a = Z[f"{comp}|{l}"]
        d.append((l, np.nanmean(a[hi]), np.nanmean(a[~hi])))
    say(f"{comp}: top-10 stand-ins, rho in high vs low dispersion:")
    for l, a, b in d:
        say(f"   {l:24s} high {a:+.3f}  low {b:+.3f}  diff {a - b:+.3f}")
a = cabz
say(f"rho(AB,Z): high {np.nanmean(a[hi]):+.3f} low {np.nanmean(a[~hi]):+.3f}")

# ---------------------------------------------------------------- ratio transfer (stated assumption)
say("\n## Ratio transfer (ASSUMPTION: the true blocker relates to each candidate like the stand-ins near it do)")
for cand in ("X", "SWD", "SWC"):
    for comp in ("AB", "Z"):
        sel = [l for l in libs if abs(cmean[f"{comp}|{l}"]) >= 0.45]
        if not sel:
            continue
        ratios = np.array([cmean[f"{cand}|{l}"] / cmean[f"{comp}|{l}"] for l in sel])
        say(f"{cand} vs {comp}-type blockers ({len(sel)} stand-ins): ratio median {np.median(ratios):.3f} "
            f"[p10 {np.percentile(ratios, 10):.3f}, p90 {np.percentile(ratios, 90):.3f}] -> predicted rho "
            f"{SERVER[comp] * np.median(ratios):.3f} [{SERVER[comp] * np.percentile(ratios, 10):.3f}, {SERVER[comp] * np.percentile(ratios, 90):.3f}]")

REPORTS.mkdir(parents=True, exist_ok=True)
(REPORTS / "c5_r06_assumption_audit.md").write_text("# C5 r06 assumption audit\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
