"""Official per-period protocol (mirrors walkforward.walkforward_full), with stricter hygiene.

Measured on the shipped data: each period file is an independent RECONSTRUCTION (universe, returns,
target and dispersion differ for the same hours across files), so history inside the latest file
carries hindsight. The honest out-of-sample unit is therefore: for period file NNN, train on its
own rows except the last 800 (minus an embargo of target_horizon+lag rows - the vendor runner has
no embargo), call generate_features on the file's full feature frame, predict the last 800 rows.

Dev periods  = all but the last `vault_months` files; vault = the last `vault_months` files.
Each fold keeps its own tickers; OOS rows are indexed "NNN:row" so folds never collide.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from lab.backtest import pnl
from lab.io import list_periods
from lab.paths import data_root

VAL_ROWS = 800


def _load(root, p):
    out = {k: pd.read_parquet(root / f"{p}.small.{k}.parquet") for k in ("features", "returns", "target")}
    dp = root / f"{p}.small.dispersion.parquet"
    out["dispersion"] = pd.read_parquet(dp)["dispersion"] if dp.exists() else None
    return out


def periods_split(s: dict, root=None) -> tuple[list[str], list[str]]:
    ps = list_periods(root or data_root())
    v = int(s["vault_months"])
    return ps[: len(ps) - v], ps[len(ps) - v:]


def run_official(cls, s: dict, periods: list[str], extra_history: int = 0, embargo: int | None = None,
                 root=None, log=print) -> list[dict]:
    """One fold per period file. extra_history > 0 also returns predictions for that many rows
    before the validation block (used by turnover research to smooth with real warm-up)."""
    root = root or data_root()
    emb = int(s["embargo_hours"]) if embargo is None else int(embargo)
    folds = []
    for p in periods:
        d = _load(root, p)
        F, T, X = d["features"], d["target"], d["returns"]
        n = len(F)
        nv = min(VAL_ROWS, int(0.2 * n))
        ntr = max(0, n - nv - emb)
        t0 = time.perf_counter()
        m = cls()
        if hasattr(m, "generate_features"):
            m.generate_features(F)
        m.train(F.iloc[:ntr], T.iloc[:ntr])
        t_train = time.perf_counter() - t0
        lo = max(0, n - nv - extra_history)
        t0 = time.perf_counter()
        P = m.predict(F.iloc[lo:])
        t_pred = time.perf_counter() - t0
        P = pd.DataFrame(P, index=F.index[lo:]).reindex(columns=X.columns)
        key = lambda idx: [f"{p}:{i}" for i in idx]  # noqa: E731
        fold = {"period": p, "n_val": nv, "train_rows": ntr, "train_s": round(t_train, 2), "predict_s": round(t_pred, 2)}
        for name, frame in (("P", P), ("X", X.iloc[lo:]), ("T", T.iloc[lo:])):
            fr = frame.copy()
            fr.index = key(fr.index)
            fold[name] = fr
        disp = d["dispersion"]
        if disp is not None:
            ds = disp.iloc[lo:].copy()
            ds.index = key(ds.index)
            fold["disp"] = ds
        fold["val_index"] = key(F.index[n - nv:])
        folds.append(fold)
        log(f"  [{p}] train_rows={ntr} train={t_train:.1f}s predict={t_pred:.1f}s")
    return folds


def val_only(folds: list[dict], P_override: dict | None = None) -> list[dict]:
    """Restrict every fold to its validation rows (optionally with replacement predictions)."""
    out = []
    for f in folds:
        vi = f["val_index"]
        P = (P_override or {}).get(f["period"], f["P"])
        g = {k: v for k, v in f.items() if k not in ("P", "X", "T", "disp")}
        g["P"], g["X"], g["T"] = P.loc[vi], f["X"].loc[vi], f["T"].loc[vi]
        if "disp" in f:
            g["disp"] = f["disp"].loc[vi]
        out.append(g)
    return out


def fold_pnl(folds: list[dict], cost_bp: float) -> pd.DataFrame:
    """PnL per fold (each fold starts flat, as the vendor backtest), concatenated."""
    return pd.concat([pnl(f["P"], f["X"], cost_bp) for f in folds])


def stacked(folds: list[dict], key: str = "P") -> pd.DataFrame:
    return pd.concat([f[key] for f in folds])
