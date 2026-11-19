"""Local emulation of the server's overfitting gate.

Server (COMPETITION.md): "We retrain your model many times on data whose labels have been shuffled
... the features are permuted row-wise and the target is recomputed from the permuted data ...
and compare your real in-sample result against that null distribution."  The vendor's
demo_engineered.py is documented to pass at p ~= 0.01 (N=100) -> calibration target.

Emulation ("server" mode, default):
  real : train on the reference slice, predict the same slice (in-sample), score.
  null : permute ROWS of (features, returns) jointly, RECOMPUTE the target from the permuted
         returns (offset-1, h-hour window sum, xs z-score, clip +-5 -- identified in r01),
         train, predict in-sample, score.
  p = (1 + #{null >= real}) / (1 + reps), for two statistics:
         net Sharpe on returns (the scored quantity) and mean IC vs the training target.
Why windowed features fail it: after a row permutation, a trailing window over random rows is a
smooth noise series, and so is the recomputed 120h target; in-sample the two fit each other
spuriously (spurious regression), which inflates the null.

"label_block" mode (legacy): permute only the target in blocks; weaker, kept for comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lab.backtest import pnl, sharpe
from lab.metrics import ic_series


def recompute_target(X: pd.DataFrame, h: int, offset: int = 1, clip: float = 5.0) -> pd.DataFrame:
    """xs z-score of sum_{k=offset}^{offset+h-1} X(i+k), clipped (look-ahead by construction)."""
    cs = X.fillna(0.0).cumsum()
    R = cs.shift(-(offset + h - 1)) - cs.shift(-offset) + X.fillna(0.0).shift(-offset)
    d = R.sub(R.mean(axis=1), axis=0)
    z = d.div(d.std(axis=1, ddof=0).replace(0, np.nan), axis=0)
    return z.clip(-clip, clip).fillna(0.0)


def _fit_score(cls, F: pd.DataFrame, T: pd.DataFrame, X: pd.DataFrame, lag: int, cost_bp: float) -> dict:
    m = cls()
    if hasattr(m, "generate_features"):
        m.generate_features(F)
    m.train(F, T)
    P = pd.DataFrame(m.predict(F), index=F.index).reindex(columns=X.columns)
    return {"sharpe": sharpe(pnl(P, X, cost_bp)["net"]), "ic": float(ic_series(P, T, lag).mean())}


def permute_rows(df: pd.DataFrame, perm: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(df.to_numpy()[perm], index=df.index, columns=df.columns)


def shuffle_gate(cls, F, T, X, lag: int, h: int, reps: int, seed: int, cost_bp: float = 5.0,
                 mode: str = "server", block: int = 120, log=print) -> dict:
    rng = np.random.default_rng(seed)
    real = _fit_score(cls, F, T, X, lag, cost_bp)
    log(f"  real   : in-sample net SR {real['sharpe']:+.5f}  IC {real['ic']:+.5f}")
    nulls = []
    for k in range(reps):
        if mode == "server":
            perm = rng.permutation(len(F))
            Fp, Xp = permute_rows(F, perm), permute_rows(X, perm)
            Tp = recompute_target(Xp, h)
        else:
            n = len(T)
            starts = np.arange(0, n, block)
            order = rng.permutation(len(starts))
            idx = np.concatenate([np.arange(starts[o], min(starts[o] + block, n)) for o in order])
            Fp, Xp, Tp = F, X, permute_rows(T, idx)
        v = _fit_score(cls, Fp, Tp, Xp, lag, cost_bp)
        nulls.append(v)
        log(f"  null {k + 1:>3}/{reps}: SR {v['sharpe']:+.5f}  IC {v['ic']:+.5f}")
    out = {"mode": mode, "reps": reps, "real_sharpe": real["sharpe"], "real_ic": real["ic"]}
    for key in ("sharpe", "ic"):
        arr = np.array([v[key] for v in nulls], dtype=float)
        arr = arr[np.isfinite(arr)]
        sd = arr.std(ddof=1) if len(arr) > 1 else float("nan")
        out[f"null_{key}_mean"] = float(arr.mean()) if len(arr) else float("nan")
        out[f"null_{key}_sd"] = float(sd)
        out[f"null_{key}_q95"] = float(np.quantile(arr, 0.95)) if len(arr) else float("nan")
        out[f"z_{key}"] = float((real[key] - arr.mean()) / sd) if sd and sd > 0 else float("nan")
        out[f"p_{key}"] = float((1 + np.sum(arr >= real[key])) / (len(arr) + 1))
    return out
