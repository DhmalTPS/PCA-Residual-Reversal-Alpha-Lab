"""Local replica of the scoring PnL.

    r(i) = <P(i-1), X(i)> - c * ||P(i-1) - P(i-2)||_1

Sharpe is invariant to a global rescaling of P (both terms are linear in P), but NOT to
time-varying scale, so the book's size path matters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def demean(P: pd.DataFrame) -> pd.DataFrame:
    return P.sub(P.mean(axis=1), axis=0)


def align(P: pd.DataFrame, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [c for c in X.columns if c in P.columns]
    idx = P.index.intersection(X.index)
    return P.loc[idx, cols].fillna(0.0), X.loc[idx, cols]


def pnl(P: pd.DataFrame, X: pd.DataFrame, cost_bp: float = 5.0) -> pd.DataFrame:
    """Per-hour gross, cost, net and turnover. Rows lacking two lags of P are dropped."""
    P, X = align(P, X)
    p1 = P.shift(1)
    p2 = P.shift(2)
    gross = (p1 * X.fillna(0.0)).sum(axis=1)
    turnover = (p1 - p2).abs().sum(axis=1)
    cost = cost_bp * 1e-4 * turnover
    out = pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost, "turnover": turnover})
    return out.iloc[2:]


def sharpe(r) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else float("nan")


def row_norms(P: pd.DataFrame, ord_: int = 2) -> pd.Series:
    A = P.fillna(0.0).to_numpy(float)
    return pd.Series(np.linalg.norm(A, ord=ord_, axis=1), index=P.index)


def flat_stats(P: pd.DataFrame, rel_threshold: float = 0.05) -> dict:
    """Fraction of 'flat' rows (norm below rel_threshold * median norm) and the longest flat run."""
    n = row_norms(P).to_numpy()
    med = np.median(n[n > 0]) if (n > 0).any() else 0.0
    flat = n <= rel_threshold * med if med > 0 else np.ones_like(n, dtype=bool)
    longest = run = 0
    for f in flat:
        run = run + 1 if f else 0
        longest = max(longest, run)
    return {"flat_fraction": float(flat.mean()) if len(flat) else 1.0, "longest_flat_run": int(longest)}
