"""Family OFFLINE_LEARNED - fixed learned weights fitted ONLY on honest (scored) rows, out-of-fold.

PRE-REGISTERED PLAN (written before any fit / evaluate call)
-----------------------------------------------------------
Pipeline of every learned candidate (identical in the OOF loop and in the shipped fn):
    U_k   = unit(rank(base_k(F)))                       K ~ 81 causal bases (levels, 1/6/24/120h changes,
                                                        |z| magnitudes, z*z products, z spreads, vols, ts-z)
    C     = sum_k w_k U_k                               (GBM: C = model prediction panel)
    S     = smooth(C, 672)                              finite exp kernel, causal
    R     = residualize(S, [smooth(S06,336), smooth(S07,672), smooth(S08,672)])   per hour, exact legs
    pos   = unit(R)                                     -> cos with the smoothed occupied legs == 0 by construction
  shipped as fn(F) -> R and scored with evaluate(fn, hl=0, warm=3500) (hl=0: harness does unit() only).

Training data: the last 800 rows of dev periods 001..029 only (the scored rows). For held-out period p the
weights are fitted on all dev periods except {p-1, p, p+1} (buffer: consecutive files overlap ~100 rows and the
120h target spans ~0.2 period). All OOF metrics replicate lab.honest.evaluate field by field.

OOF configurations (each logged as ONE trial in the registry, mode 'screen', data 'real-official'):
    O1 mv_l0.1   mean-variance on factor PnL g_k(t)=<R_k(t-1),X(t)>, ridge lam=0.1*tr(S)/K
    O2 mv_l1     same, lam=1
    O3 mv_l10    same, lam=10
    O4 pca5      MV restricted to the top-5 principal components of the factor-PnL covariance
    O5 icridge   pooled ridge of zx(T(t)) on R_k(t-1) (all hours x tickers), lam=1*tr/K
    O6 signcons  MV lam=1 on bases whose training per-period mean PnL has the same sign in >=70% of periods
    O7 diag_sc   diagonal MV (mu_k/var_k) on the same sign-consistent subset
    O8 gbm_d2    LightGBM depth 2, 300 trees, lr 0.03, min_data 4000, on unit-rank bases -> xs rank of X(t+1)
evaluate() calls (HARD LIMIT 12; planned <= 4): the full-dev-fit fixed weights of the best <= 3 OOF configs
(in-sample by construction for the weights -> reported as such; the OOF numbers are the honest ones), plus one
spare. If no OOF config has SR5 > 0.008, at most 1 evaluate call (best config) and report a null.

Stages:  python research/agents/offline_learned.py cache | oof | final <config...>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from lab import honest as H  # noqa: E402
from lab import registry  # noqa: E402
from lab.backtest import pnl, sharpe  # noqa: E402
from lab.metrics import bootstrap_stat, deflated_sharpe, ic_series, mean_se_nw, row_cos, sharpe_se_nw  # noqa: E402
from lab.paths import settings  # noqa: E402

OUT = Path("research/agents/_offline_learned")
OUT.mkdir(parents=True, exist_ok=True)
VAL, WARM, HL = 800, 3500, 672.0


# ----------------------------------------------------------------------------- causal bases
def base_panels(F: pd.DataFrame):
    """Generator of (name, causal raw panel) - lazy so the full-file gate run stays memory-light."""
    g = H.blocks(F)
    z = {i: H.zx(g[i]) for i in g}
    for i in range(1, 7):
        yield f"lvl{i}", g[i]
        for k in (1, 6, 24, 120):
            yield f"d{k}_{i}", g[i].diff(k)
        yield f"az0_{i}", z[i].abs()
        yield f"az24_{i}", H.zx(g[i].diff(24)).abs()
        yield f"az120_{i}", H.zx(g[i].diff(120)).abs()
        yield f"zxd24_{i}", z[i] * H.zx(g[i].diff(24))
        d1 = g[i].diff(1)
        yield f"vol24_{i}", d1.rolling(24, min_periods=12).std()
        yield f"vol168_{i}", d1.rolling(168, min_periods=84).std()
        m = g[i].rolling(168, min_periods=84)
        yield f"tsz_{i}", (g[i] - m.mean()) / m.std()
    for i in (1, 2, 3, 4):
        yield f"sp{i}_6", z[i] - z[6]
        yield f"sp{i}_5", z[i] - z[5]
    yield "sp5_6", z[5] - z[6]


def iter_unit_bases(F: pd.DataFrame, cols):
    for k, v in base_panels(F):
        yield k, H.unit(H.rank(v.reindex(columns=cols)))


def unit_bases(F: pd.DataFrame, cols) -> dict[str, pd.DataFrame]:
    return dict(iter_unit_bases(F, cols))


def smoothed_legs(F: pd.DataFrame, cols) -> list[pd.DataFrame]:
    raw = H.occupied_raw(F)
    return [H.smooth(raw[k].reindex(columns=cols), H.LEG_HL[k]) for k in ("S06", "S07", "S08")]


def project_out(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A: (..., T, N) demeaned; B: (T, N, 3) demeaned -> per-hour LS residual."""
    Pi = np.linalg.pinv(B)  # (T,3,N)
    coef = np.einsum("tkn,...tn->...tk", Pi, A)
    return A - np.einsum("tnk,...tk->...tn", B, coef)


# ----------------------------------------------------------------------------- stage 1: cache
def build_cache():
    for p in H.dev_periods():
        f = OUT / f"c_{p}.npz"
        if f.exists():
            continue
        D = H._load(p, VAL + WARM)
        F, X, T = D["features"], D["returns"], D["target"]
        cols = list(X.columns)
        UB = unit_bases(F, cols)
        names = list(UB)
        legs = smoothed_legs(F, cols)
        Bv = np.stack([l.iloc[-VAL:].to_numpy(float) for l in legs], -1)
        Bv = Bv - Bv.mean(1, keepdims=True)
        S = np.stack([H.smooth(UB[k], HL).iloc[-VAL:].to_numpy(float) for k in names])  # K,T,N
        S = S - S.mean(2, keepdims=True)
        R = project_out(S, Bv).astype(np.float32)
        Uv = np.stack([UB[k].iloc[-VAL:].to_numpy(float) for k in names]).astype(np.float32)
        np.savez(f, R=R, U=Uv, X=X.iloc[-VAL:].to_numpy(float), T=T.iloc[-VAL:].to_numpy(float),
                 B=Bv.astype(np.float32), names=np.array(names), cols=np.array(cols))
        print(p, R.shape, flush=True)


def load_cache():
    C = {}
    for p in H.dev_periods():
        d = np.load(OUT / f"c_{p}.npz")
        C[p] = {k: d[k] for k in d.files}
    return C


# ----------------------------------------------------------------------------- weight learners
def factor_pnl(R, X):
    """g_k(t) = <R_k(t-1), X(t)> for t=2..T-1 (pnl drops the first two rows)."""
    Xn = np.nan_to_num(X)
    return np.einsum("ktn,tn->tk", R[:, 1:-1], Xn[2:])


def fit(cfg, C, train):
    K = next(iter(C.values()))["R"].shape[0]
    if cfg == "gbm_d2":
        return fit_gbm(C, train)
    if cfg == "icridge":
        A = np.zeros((K, K)); b = np.zeros(K)
        for p in train:
            R, T = C[p]["R"].astype(float), C[p]["T"]
            Tz = T - np.nanmean(T, 1, keepdims=True)
            Tz = np.nan_to_num(Tz / np.nanstd(Tz, 1, keepdims=True))
            M = R[:, :-1].reshape(K, -1)
            A += M @ M.T; b += M @ Tz[1:].reshape(-1)
        lam = np.trace(A) / K
        return np.linalg.solve(A + lam * np.eye(K), b)
    G = [factor_pnl(C[p]["R"].astype(float), C[p]["X"]) for p in train]
    Gall = np.concatenate(G)
    mu, Sig = Gall.mean(0), np.cov(Gall, rowvar=False)
    tr = np.trace(Sig) / K
    keep = np.ones(K, bool)
    if cfg in ("signcons", "diag_sc"):
        pm = np.stack([g.mean(0) for g in G])
        fp = (pm > 0).mean(0)
        keep = (fp >= 0.7) | (fp <= 0.3)
        if not keep.any():
            return np.zeros(K)
    w = np.zeros(K)
    if cfg.startswith("mv_l") or cfg == "signcons":
        lam = 1.0 if cfg == "signcons" else float(cfg[4:])
        idx = np.where(keep)[0]
        w[idx] = np.linalg.solve(Sig[np.ix_(idx, idx)] + lam * tr * np.eye(len(idx)), mu[idx])
    elif cfg == "diag_sc":
        w[keep] = mu[keep] / np.diag(Sig)[keep]
    elif cfg == "pca5":
        ev, V = np.linalg.eigh(Sig)
        V, ev = V[:, -5:], ev[-5:]
        w = V @ ((V.T @ mu) / ev)
    return w


def fit_gbm(C, train):
    import lightgbm as lgb
    Xs, ys = [], []
    for p in train:
        U, X = C[p]["U"], C[p]["X"]
        y = pd.DataFrame(X[1:]).rank(axis=1, pct=True).to_numpy() - 0.5   # rank of X(t+1)
        Xs.append(U[:, :-1].reshape(U.shape[0], -1).T); ys.append(y.reshape(-1))
    Xm, y = np.concatenate(Xs), np.concatenate(ys)
    ok = np.isfinite(y)
    m = lgb.LGBMRegressor(max_depth=2, num_leaves=3, n_estimators=300, learning_rate=0.03,
                          min_child_samples=4000, subsample=0.7, subsample_freq=1, colsample_bytree=0.7,
                          reg_lambda=10.0, verbose=-1, random_state=7)
    m.fit(Xm[ok], y[ok])
    return m


def gbm_R(model, p):
    """Held-out GBM position: predict on all rows of the file, then smooth + residualise like the fn."""
    D = H._load(p, VAL + WARM)
    F, X = D["features"], D["returns"]
    cols = list(X.columns)
    UB = unit_bases(F, cols)
    A = np.stack([UB[k].to_numpy(float) for k in UB], -1)  # T,N,K
    pr = model.predict(np.nan_to_num(A.reshape(-1, A.shape[-1]))).reshape(A.shape[:2])
    P = H.unit(pd.DataFrame(pr, index=F.index, columns=cols))
    S = H.smooth(P, HL).iloc[-VAL:].to_numpy(float)
    legs = smoothed_legs(F, cols)
    Bv = np.stack([l.iloc[-VAL:].to_numpy(float) for l in legs], -1)
    Bv = Bv - Bv.mean(1, keepdims=True)
    return project_out(S - S.mean(1, keepdims=True), Bv)


# ----------------------------------------------------------------------------- OOF metrics (= evaluate fields)
def metrics(pos: dict, C, name, log=True):
    s = settings(); lag = int(s["nw_lag_hours"])
    acc5, acc10, ics, per, cos = [], [], [], [], {"S06": [], "S07": [], "S08": []}
    for p, Rp in pos.items():
        cols = [str(c) for c in C[p]["cols"]]
        idx = pd.RangeIndex(VAL)
        U = H.unit(pd.DataFrame(Rp, index=idx, columns=cols))
        X = pd.DataFrame(C[p]["X"], index=idx, columns=cols)
        T = pd.DataFrame(C[p]["T"], index=idx, columns=cols)
        p5 = pnl(U, X, 5.0)["net"].reset_index(drop=True)
        acc5.append(p5); per.append(sharpe(p5))
        acc10.append(pnl(U, X, 10.0)["net"].reset_index(drop=True))
        ics.append(ic_series(U, T, 1).dropna().reset_index(drop=True))
        legs = H.occupied_smoothed(p, WARM)
        for k in cos:
            L = legs[k].reset_index(drop=True); L.columns = [str(c) for c in L.columns]
            cos[k].append(float(row_cos(U, L).mean()))
    r5 = pd.concat(acc5, ignore_index=True)
    sr5, se5, vr = sharpe_se_nw(r5, lag)
    sr10, _, _ = sharpe_se_nw(pd.concat(acc10, ignore_index=True), lag)
    ci = bootstrap_stat(r5.to_numpy(), "sharpe", float(s["bootstrap_block_hours"]), 500, int(s["seed"]))
    rho = pd.concat(ics, ignore_index=True)
    ic, ise = mean_se_nw(rho.to_numpy(), lag)
    out = {"name": name, "hl": HL, "SR5": sr5, "se": se5, "ci_lo": ci["lo"], "ci_hi": ci["hi"], "SR10": sr10,
           "periods_pos": float(np.mean(np.array(per) > 0)), "IC_T": ic, "IC_t": ic / ise if ise else np.nan,
           "cos06": float(np.mean(cos["S06"])), "cos07": float(np.mean(cos["S07"])), "cos08": float(np.mean(cos["S08"]))}
    if log:
        registry.append({"run_id": f"oof_{name}", "utc": pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%S"),
                         "strategy": "agents.offline_learned", "mode": "screen", "data": "real-official", "IC": ic,
                         "IC_t": out["IC_t"], "SR_net_5bp": sr5, "SR_se_5bp": se5, "SR_net_10bp": sr10,
                         "overrides": name, "notes": "OOF leave-period-out (+-1 buffer) learned weights"})
    n = registry.n_trials("real-official")
    t_eff = len(r5) / vr if np.isfinite(vr) and vr > 0 else len(r5)
    out["SR_star"], out["DSR"] = deflated_sharpe(sr5, t_eff, 0.0, 3.0, n, se5)
    out["n_trials"] = n
    return out


def run_oof(cfgs):
    C = load_cache()
    ps = H.dev_periods()
    names = [str(x) for x in C[ps[0]]["names"]]
    res = []
    for cfg in cfgs:
        pos, W = {}, []
        for j, p in enumerate(ps):
            train = [q for i, q in enumerate(ps) if abs(i - j) > 1]
            w = fit(cfg, C, train)
            if cfg == "gbm_d2":
                pos[p] = gbm_R(w, p)
            else:
                W.append(w)
                pos[p] = np.einsum("k,ktn->tn", w, C[p]["R"].astype(float))
        m = metrics(pos, C, f"ol_{cfg}")
        if W:
            Wa = np.stack(W); Wn = Wa / np.linalg.norm(Wa, axis=1, keepdims=True)
            m["fold_weight_cos_mean"] = float((Wn @ Wn.T)[np.triu_indices(len(Wn), 1)].mean())
            wf = fit(cfg, C, ps)
            top = np.argsort(-np.abs(wf))[:8]
            m["top_full_weights"] = {names[i]: round(float(wf[i] / np.abs(wf).max()), 3) for i in top}
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}), flush=True)
        res.append(m)
    pd.DataFrame(res).to_csv(OUT / "oof_results.csv", mode="a", index=False,
                             header=not (OUT / "oof_results.csv").exists())


# ----------------------------------------------------------------------------- stage 3: fixed-weight fn + evaluate
def make_fn(names, w):
    wd = {n: float(x) for n, x in zip(names, w) if x != 0}

    def fn(F):
        cols = list(dict.fromkeys(F.columns.get_level_values(-1)))
        Cc = 0.0
        for k, U in iter_unit_bases(F, cols):
            if k in wd:
                Cc = Cc + wd[k] * U
        S = H.smooth(Cc, HL)
        return H.residualize(S, smoothed_legs(F, cols))
    return fn


def run_final(cfgs):
    C = load_cache()
    ps = H.dev_periods()
    names = [str(x) for x in C[ps[0]]["names"]]
    for cfg in cfgs:
        w = fit(cfg, C, ps)
        json.dump(dict(zip(names, map(float, w))), open(OUT / f"weights_{cfg}.json", "w"), indent=1)
        H.evaluate(make_fn(names, w), f"ol_full_{cfg}", hl=0, warm=WARM,
                   notes="offline_learned full-dev fixed weights (weights in-sample on dev)")


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "cache":
        build_cache()
    elif st == "oof":
        run_oof(sys.argv[2:])
    elif st == "final":
        run_final(sys.argv[2:])


# ============================================================================= STAGE 2 (pre-registered after stage-1 OOF)
# Stage-1 OOF (8 trials): best pca5 SR5 .0134 / DSR .24; weights stable across folds (cos .92-.97) and concentrated on
# F1-4 near-collinear copies. Stage-2 plan (4 OOF trials, <= 2 more evaluate calls):
#   G1 g_mv_l1    grouped bases (F1-4 averaged per type -> 39 groups), MV ridge lam=1
#   G2 g_pca5     grouped, PCA-5 MV
#   G3 g_nested   grouped, per outer fold choose among {mv .1,1,10, pca3,5,8} by INNER leave-period-out (+-1 buffer)
#                 Sharpe -> removes the best-of-configs selection bias
#   G4 g_pca5@1344  as G2 with smoothing half-life 1344 (turnover / SR10)
TYPES = ["lvl", "d1", "d6", "d24", "d120", "az0", "az24", "az120", "zxd24", "vol24", "vol168", "tsz"]


def group_matrix(names):
    rows, gn = [], []
    ix = {n: i for i, n in enumerate(names)}
    def key(t, i):
        return f"{t}{i}" if t == "lvl" else f"{t}_{i}"
    for t in TYPES:
        for lab, members in (("14", (1, 2, 3, 4)), ("5", (5,)), ("6", (6,))):
            r = np.zeros(len(names))
            for i in members:
                r[ix[key(t, i)]] = 1.0 / len(members)
            rows.append(r); gn.append(f"{t}_{lab}")
    for lab, mem in (("sp14_6", [f"sp{i}_6" for i in (1, 2, 3, 4)]), ("sp14_5", [f"sp{i}_5" for i in (1, 2, 3, 4)]),
                     ("sp5_6", ["sp5_6"])):
        r = np.zeros(len(names))
        for m in mem:
            r[ix[m]] = 1.0 / len(mem)
        rows.append(r); gn.append(lab)
    return np.array(rows), gn


def grouped_cache(C, M):
    return {p: {**d, "R": np.einsum("gk,ktn->gtn", M, d["R"].astype(float))} for p, d in C.items()}


def build_cache_hl(hl):
    """Re-cache R at another smoothing half-life (only R, X, T, cols, names)."""
    for p in H.dev_periods():
        f = OUT / f"c{int(hl)}_{p}.npz"
        if f.exists():
            continue
        D = H._load(p, VAL + int(4 * hl) + 300)
        F, X, T = D["features"], D["returns"], D["target"]
        cols = list(X.columns)
        legs = H.occupied_smoothed(p, WARM)
        Bv = np.stack([legs[k].reindex(columns=cols).to_numpy(float) for k in ("S06", "S07", "S08")], -1)
        Bv = Bv - Bv.mean(1, keepdims=True)
        names, S = [], []
        for k, U in iter_unit_bases(F, cols):
            names.append(k); S.append(H.smooth(U, hl).iloc[-VAL:].to_numpy(float))
        S = np.stack(S); S = S - S.mean(2, keepdims=True)
        np.savez(f, R=project_out(S, Bv).astype(np.float32), X=X.iloc[-VAL:].to_numpy(float),
                 T=T.iloc[-VAL:].to_numpy(float), names=np.array(names), cols=np.array(cols))
        print(p, flush=True)


def fit_from_G(cfg, G):
    Gall = np.concatenate(G)
    mu, Sig = Gall.mean(0), np.cov(Gall, rowvar=False)
    K = len(mu); tr = np.trace(Sig) / K
    if cfg.startswith("mv"):
        return np.linalg.solve(Sig + float(cfg[2:]) * tr * np.eye(K), mu)
    n = int(cfg[3:])
    ev, V = np.linalg.eigh(Sig)
    V, ev = V[:, -n:], ev[-n:]
    return V @ ((V.T @ mu) / ev)


SUBS = ["mv0.1", "mv1", "mv10", "pca3", "pca5", "pca8"]


def nested_fit(Gd, train):
    best, bs = None, -np.inf
    for c in SUBS:
        r = []
        for j, q in enumerate(train):
            inner = [x for i, x in enumerate(train) if abs(i - j) > 1]
            w = fit_from_G(c, [Gd[x] for x in inner])
            r.append(Gd[q] @ w / max(np.linalg.norm(w), 1e-12))
        r = np.concatenate(r); s = r.mean() / r.std()
        if s > bs:
            best, bs = c, s
    return best, fit_from_G(best, [Gd[x] for x in train])


def run_oof2(cfgs):
    ps = H.dev_periods()
    for cfg in cfgs:
        base, _, hl = cfg.partition("@")
        if hl:
            build_cache_hl(float(hl))
            C = {}
            for p in ps:
                d = np.load(OUT / f"c{int(float(hl))}_{p}.npz"); C[p] = {k: d[k] for k in d.files}
        else:
            C = load_cache()
        names = [str(x) for x in C[ps[0]]["names"]]
        M, gn = group_matrix(names)
        Cg = grouped_cache(C, M)
        Gd = {p: factor_pnl(Cg[p]["R"], Cg[p]["X"]) for p in ps}
        pos, W, picks = {}, [], []
        for j, p in enumerate(ps):
            train = [q for i, q in enumerate(ps) if abs(i - j) > 1]
            if base == "g_nested":
                c, w = nested_fit(Gd, train); picks.append(c)
            else:
                w = fit_from_G({"g_mv_l1": "mv1", "g_pca5": "pca5"}[base], [Gd[q] for q in train])
            W.append(w)
            pos[p] = np.einsum("g,gtn->tn", w, Cg[p]["R"])
        m = metrics(pos, C, f"ol_{cfg}")
        Wa = np.stack(W); Wn = Wa / np.linalg.norm(Wa, axis=1, keepdims=True)
        m["fold_weight_cos_mean"] = float((Wn @ Wn.T)[np.triu_indices(len(Wn), 1)].mean())
        if picks:
            m["nested_picks"] = pd.Series(picks).value_counts().to_dict()
            wf = nested_fit(Gd, ps)[1]
        else:
            wf = fit_from_G({"g_mv_l1": "mv1", "g_pca5": "pca5"}[base], [Gd[q] for q in ps])
        m["full_weights_grouped"] = {gn[i]: round(float(wf[i] / np.abs(wf).max()), 3) for i in np.argsort(-np.abs(wf))[:10]}
        json.dump({"groups": gn, "w_group": wf.tolist(), "w_base": dict(zip(names, (M.T @ wf).tolist()))},
                  open(OUT / f"weights_{cfg.replace('@', '_')}.json", "w"), indent=1)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}), flush=True)


if __name__ == "__main__" and sys.argv[1] == "oof2":
    run_oof2(sys.argv[2:])


# ----------------------------------------------------------------------------- stage 2 final (evaluate call #3)
def make_fn_hl(wd: dict, hl: float):
    wd = {k: float(v) for k, v in wd.items() if abs(v) > 0}

    def fn(F):
        cols = list(dict.fromkeys(F.columns.get_level_values(-1)))
        Cc = 0.0
        for k, U in iter_unit_bases(F, cols):
            if k in wd:
                Cc = Cc + wd[k] * U
        return H.residualize(H.smooth(Cc, hl), smoothed_legs(F, cols))
    return fn


if __name__ == "__main__" and sys.argv[1] == "final2":
    cfg, hl = sys.argv[2], float(sys.argv[3])
    wd = json.load(open(OUT / f"weights_{cfg}.json"))["w_base"]
    H.evaluate(make_fn_hl(wd, hl), f"ol_full_{cfg}", hl=0, warm=int(4 * hl) + 400,
               notes="offline_learned grouped full-dev fixed weights (weights in-sample on dev)")
