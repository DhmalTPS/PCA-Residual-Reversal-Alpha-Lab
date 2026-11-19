"""Shared HONEST candidate-evaluation harness (used by every family search - one protocol for all).

evaluate(fn, name) scores a candidate construction exactly the way submissions are scored, plus the
novelty and gate diagnostics that decided C4 so far:

  fn(F) -> DataFrame (rows x tickers): a CAUSAL raw score computed from one period file's feature frame F
          (any number of rows; only hours <= t may be used for row t; sign = intended direction).

  protocol  : for each dev period file, compute fn on that file's rows (warm-up included), unit-normalise
              per hour, smooth (finite exponential kernel, half-life hl), score the last 800 rows only.
  SR5/SR10  : net Sharpe on the returns panel at 5/10 bp, NW s.e. (lag 48), 90% stationary-bootstrap CI.
  IC_T      : mean per-hour IC vs the target (NW t).
  periods+  : share of the 29 dev periods with positive net Sharpe.
  cosNN     : mean per-hour cosine of the smoothed candidate with each smoothed occupied leg defined by the
              ACTIVE SEASON in seasons/<S>/occupied_directions.py (S1: S06, S07, S08, S09).
  gate_inSR : in-sample net Sharpe over the gate slice (training rows of the latest dev file); fixed-signal null
              mean ~ -0.005 to -0.007 (our emulation is known to be too strict for reversal-type signals).
  DSR       : deflated Sharpe with the registry's honest trial count (every evaluate() call is a trial).

Memory-light: one period file at a time, rows trimmed to 800 + warm-up. Occupied legs are cached on disk.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from lab import registry
from lab.backtest import pnl, sharpe
from lab.metrics import bootstrap_stat, deflated_sharpe, ic_series, mean_se_nw, row_cos, sharpe_se_nw
from lab.official import periods_split
from lab.paths import EXPERIMENTS, data_root, settings

CACHE = EXPERIMENTS / "cache"
VAL = 800


# ----------------------------------------------------------------------------- primitives (research side)
def rank(df: pd.DataFrame) -> pd.DataFrame:
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).to_numpy()[:, None]
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))


def zx(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sub(df.mean(axis=1), axis=0)
    return d.div(d.std(axis=1).replace(0, np.nan), axis=0)


def unit(S: pd.DataFrame) -> pd.DataFrame:
    d = S.sub(S.mean(axis=1), axis=0).fillna(0.0)
    n = np.sqrt((d * d).sum(axis=1))
    return d.div(n.where(n > 0), axis=0).fillna(0.0)


def smooth(S: pd.DataFrame, hl: float) -> pd.DataFrame:
    if hl <= 0:
        return S.sub(S.mean(axis=1), axis=0)
    W = int(4 * hl)
    w = 0.5 ** (np.arange(W) / hl)
    A = S.fillna(0.0).to_numpy(float)
    out = pd.DataFrame(lfilter(w, [1.0], A, axis=0) / lfilter(w, [1.0], np.ones((len(A), 1)), axis=0),
                       index=S.index, columns=S.columns)
    return out.sub(out.mean(axis=1), axis=0)


def residualize(C: pd.DataFrame, bases: list[pd.DataFrame]) -> pd.DataFrame:
    """Per-hour least-squares residual of C on the given cross-sectional bases (all de-meaned)."""
    A = C.fillna(0.0).to_numpy(float)
    A = A - A.mean(axis=1, keepdims=True)
    B = np.stack([b.reindex_like(C).fillna(0.0).to_numpy(float) for b in bases], axis=-1)
    B = B - B.mean(axis=1, keepdims=True)
    coef = np.einsum("tkj,tj->tk", np.linalg.pinv(B), A)
    return pd.DataFrame(A - np.einsum("tjk,tk->tj", B, coef), index=C.index, columns=C.columns)


def blocks(F: pd.DataFrame) -> dict[int, pd.DataFrame]:
    tick = list(dict.fromkeys(F.columns.get_level_values(-1)))
    return {i: F[f"Feature.{i}"].reindex(columns=tick).astype(float) for i in range(1, 7)}


def _season_module():
    """Load seasons/<SEASON>/occupied_directions.py of the ACTIVE season (None if the season has none yet)."""
    import importlib.util

    from lab.paths import SEASON

    f = SEASON / "occupied_directions.py"
    if not f.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"occupied_{SEASON.name}", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def occupied_raw(F: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Unsmoothed unit scores of the season's occupied directions (seasons/<S>/occupied_directions.py)."""
    mod = _season_module()
    return mod.occupied_raw(F) if mod else {}


def leg_hl() -> dict[str, float]:
    mod = _season_module()
    return dict(mod.LEG_HL) if mod else {}


def server_angles() -> dict[str, float]:
    mod = _season_module()
    return dict(mod.SERVER_ANGLES) if mod else {}


# ----------------------------------------------------------------------------- data access
def _load(p: str, rows: int | None = None) -> dict:
    root = data_root()
    out = {k: pd.read_parquet(root / f"{p}.small.{k}.parquet") for k in ("features", "returns", "target")}
    if rows:
        n = len(out["features"])
        out = {k: v.iloc[max(0, n - rows):] for k, v in out.items()}
    return out


def dev_periods() -> list[str]:
    return periods_split(settings())[0]


def occupied_smoothed(p: str, warm: int) -> dict[str, pd.DataFrame]:
    """Smoothed occupied legs on the 800 validation rows of period p (disk-cached; cache key = the leg set)."""
    hl = leg_hl()
    if not hl:
        return {}
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"legs_{'-'.join(sorted(hl))}_{p}.parquet"
    if f.exists():
        d = pd.read_parquet(f)
        return {k: d[k] for k in hl}
    D = _load(p, VAL + max(warm, 4 * int(max(hl.values())) + 300))
    raw = occupied_raw(D["features"])
    vi = D["features"].index[len(D["features"]) - VAL:]
    legs = {k: smooth(v, hl[k]).loc[vi] for k, v in raw.items()}
    pd.concat(legs, axis=1).to_parquet(f)
    return legs


# ----------------------------------------------------------------------------- evaluation
def _fold_scores(fn, p: str, hl: float, warm: int):
    D = _load(p, VAL + warm)
    F, X, T = D["features"], D["returns"], D["target"]
    vi = F.index[len(F) - VAL:]
    U = unit(fn(F).reindex(index=F.index, columns=X.columns))
    Sm = smooth(U, hl).loc[vi]
    return Sm, U, X.loc[vi], T, vi


def evaluate(fn, name: str, hl: float = 672.0, warm: int | None = None, log_trial: bool = True,
             notes: str = "", gate: bool = True, verbose: bool = True, recent_k: int = 8, vault: bool = True) -> dict:
    s = settings()
    lag = int(s["nw_lag_hours"])
    warm = int(warm if warm is not None else 4 * hl + 600)
    acc5, acc10, ics = [], [], []
    cos = {k: [] for k in leg_hl()}
    per = []
    devp = dev_periods()
    for p in devp:
        D = _load(p, VAL + warm)
        F, X, T = D["features"], D["returns"], D["target"]
        vi = F.index[len(F) - VAL:]
        U = unit(fn(F).reindex(index=F.index, columns=X.columns))
        Sm = smooth(U, hl).loc[vi]
        p5 = pnl(Sm, X.loc[vi], 5.0)["net"].reset_index(drop=True)
        acc5.append(p5)
        acc10.append(pnl(Sm, X.loc[vi], 10.0)["net"].reset_index(drop=True))
        ics.append(ic_series(U, T, 1).reindex(vi).dropna().reset_index(drop=True))
        per.append(sharpe(p5))
        legs = occupied_smoothed(p, warm)
        for k in cos:
            cos[k].append(float(row_cos(Sm, legs[k].reindex(index=Sm.index, columns=Sm.columns)).mean()))
    r5 = pd.concat(acc5, ignore_index=True)
    sr5, se5, vr = sharpe_se_nw(r5, lag)
    sr10, _, _ = sharpe_se_nw(pd.concat(acc10, ignore_index=True), lag)
    ci = bootstrap_stat(r5.to_numpy(), "sharpe", float(s["bootstrap_block_hours"]), 500, int(s["seed"]))
    rho = pd.concat(ics, ignore_index=True)
    ic, ise = mean_se_nw(rho.to_numpy(), lag)
    recent = pd.concat(acc5[len(acc5) - recent_k:], ignore_index=True) if recent_k else pd.Series(dtype=float)
    sr_recent, se_recent, _ = sharpe_se_nw(recent, lag) if len(recent) else (float("nan"), float("nan"), float("nan"))
    out = {"name": name, "hl": hl, "SR5": sr5, "se": se5, "ci_lo": ci["lo"], "ci_hi": ci["hi"], "SR10": sr10,
           "periods_pos": float(np.mean(np.array(per) > 0)), "IC_T": ic, "IC_t": ic / ise if ise else float("nan"),
           **{f"cos{k[1:] if k[:1] == 'S' else k}": float(np.mean(v)) for k, v in cos.items()}}
    out["SR5_recent"] = sr_recent
    out["se_recent"] = se_recent
    out["recent_periods"] = f"{devp[len(devp) - recent_k]}..{devp[-1]}" if recent_k and len(devp) >= recent_k else ""
    if vault:
        vr5, vic, vicp = [], [], []
        for p in periods_split(settings())[1]:
            Sm, U, Xv, T, vi = _fold_scores(fn, p, hl, warm)
            vr5.append(pnl(Sm, Xv, 5.0)["net"].reset_index(drop=True))
            vic.append(ic_series(U, T, 1).reindex(vi).dropna().reset_index(drop=True))
            Tv = T.loc[vi]
            vicp.append(ic_series(Sm, Tv, 1).dropna().reset_index(drop=True))
        if vr5:
            proxy = pd.concat(acc5[len(acc5) - recent_k:] + vr5, ignore_index=True) if recent_k else pd.concat(vr5, ignore_index=True)
            psr, pse, _ = sharpe_se_nw(proxy, lag)
            out.update({"SR5_proxy": psr, "se_proxy": pse,
                        "proxy_periods_pos": float(np.mean([sharpe(x) > 0 for x in acc5[len(acc5) - recent_k:] + vr5]))})
            vsr, vse, _ = sharpe_se_nw(pd.concat(vr5, ignore_index=True), lag)
            a, ase = mean_se_nw(pd.concat(vic, ignore_index=True).to_numpy(), lag)
            b, bse = mean_se_nw(pd.concat(vicp, ignore_index=True).to_numpy(), lag)
            out.update({"vault_SR5": vsr, "vault_se": vse, "vault_IC_raw": a, "vault_IC_raw_t": a / ase if ase else float("nan"),
                        "vault_IC_pos": b, "vault_IC_pos_t": b / bse if bse else float("nan")})
    if gate:
        ref = dev_periods()[-1]
        D = _load(ref)
        F, X = D["features"], D["returns"]
        tr = F.index[: len(F) - VAL]
        U = unit(fn(F).reindex(index=F.index, columns=X.columns))
        out["gate_inSR"] = sharpe(pnl(smooth(U, hl).loc[tr], X.loc[tr], 5.0)["net"])
    if log_trial:
        registry.append({"run_id": f"honest_{name}", "utc": pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%S"),
                         "strategy": "lab.honest", "mode": "screen", "data": "real-official", "IC": ic,
                         "IC_t": out["IC_t"], "SR_net_5bp": sr5, "SR_se_5bp": se5, "SR_net_10bp": sr10,
                         "overrides": name, "notes": notes or "lab.honest.evaluate"})
    n_trials = registry.n_trials("real-official")
    t_eff = len(r5) / vr if np.isfinite(vr) and vr > 0 else len(r5)
    out["SR_star"], out["DSR"] = deflated_sharpe(sr5, t_eff, 0.0, 3.0, n_trials, se5)
    out["n_trials"] = n_trials
    # worst-case (triangle-bound) angle to each known occupant
    occ = {f"cos{k[1:] if k[:1] == 'S' else k}": d for k, d in server_angles().items()}
    out["min_deg_to_occupants"] = (float(min(math.degrees(math.acos(max(-1.0, min(1.0, out[k])))) - d for k, d in occ.items() if k in out))
                                   if occ else float("nan"))
    if verbose:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}))
    return out


def admissible_v2(m: dict) -> bool:
    """PRE-REGISTERED bar from S1/C5 on (lesson of S09: dev +0.026 but vault IC t -2.5 and board 0.0028):
    the v1 bar AND positive net Sharpe on the most recent dev periods AND non-negative vault net Sharpe (the scored
    quantity). vault_IC_raw / vault_IC_pos (vs the 120h target) are diagnostics only: positions smoothed at a different
    horizon can earn on returns while their IC against the 120h target is negative."""
    return (admissible(m) and m.get("SR5_recent", float("nan")) > 0
            and m.get("vault_SR5", float("nan")) >= 0)


def admissible(m: dict) -> bool:
    """Pre-registered bar for spending a server novelty check (see STATUS.md, session 3)."""
    return (m["SR5"] >= 0.0133 and m["ci_lo"] > 0 and m["SR10"] > 0.008 and m["periods_pos"] >= 0.65
            and m["DSR"] >= 0.5 and max([abs(v) for k, v in m.items() if k.startswith("cos")] or [0.0]) <= 0.35)


def __getattr__(name):
    """Backward compatibility for S1/C4 archive scripts: `from lab.honest import LEG_HL` -> active season's half-lives."""
    if name == "LEG_HL":
        return leg_hl()
    raise AttributeError(name)
