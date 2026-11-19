"""Signal geometry: unit trajectories, rho-hat between signals, gauge-frame city (best effort).

Admission rule (official):   |rho_hat(a, b)| <= 0.5, rho_hat = mean_i <u_a(i), u_b(i)>.
If the per-hour gauge map is orthonormal, rho_hat is gauge-invariant, so it can be computed
in the ticker frame of the latest period. City / concentration need the gauge frame itself:
the shipped gauge matrices are applied if their shape can be recognised; otherwise defer to
`runner.py --gauge-fix`, which is authoritative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lab.metrics import bootstrap_stat, row_cos


def rho_hat(A: pd.DataFrame, B: pd.DataFrame, block: float = 72, reps: int = 500, seed: int = 0) -> dict:
    c = row_cos(A, B).dropna()
    if len(c) < 10:
        return {"rho": float("nan"), "lo": float("nan"), "hi": float("nan"), "abs_ub": float("nan"), "hours": len(c)}
    ci = bootstrap_stat(c.to_numpy(), "mean", block, reps, seed)
    return {
        "rho": float(c.mean()),
        "lo": ci["lo"],
        "hi": ci["hi"],
        "abs_ub": float(max(abs(ci["lo"]), abs(ci["hi"]))),
        "hours": int(len(c)),
    }


def describe_gauge(panel) -> dict:
    out = {}
    for name, path in panel.gauge.items():
        try:
            a = np.load(path, mmap_mode="r", allow_pickle=False)
            out[name] = tuple(a.shape)
        except Exception as e:  # noqa: BLE001 - descriptive only
            out[name] = f"unreadable: {e}"
    return out


def gauge_fixed(P: pd.DataFrame, panel) -> np.ndarray | None:
    """Return s~(i) (T x (J-1)) if a per-hour (T, J-1, J) or (T, J, J-1) gauge array is found."""
    T, J = len(panel.index), len(panel.tickers)
    A = P.reindex(index=panel.index, columns=panel.tickers).fillna(0.0).to_numpy(float)
    A = A - A.mean(axis=1, keepdims=True)
    n = np.linalg.norm(A, axis=1, keepdims=True)
    U = np.divide(A, n, out=np.zeros_like(A), where=n > 0)
    for path in panel.gauge.values():
        try:
            G = np.load(path, mmap_mode="r", allow_pickle=False)
        except Exception:  # noqa: BLE001
            continue
        if G.ndim == 3 and G.shape[0] == T and G.shape[1:] == (J - 1, J):
            return np.einsum("tkj,tj->tk", G, U)
        if G.ndim == 3 and G.shape[0] == T and G.shape[1:] == (J, J - 1):
            return np.einsum("tjk,tj->tk", G, U)
    return None


def city_stats(S: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Concentration ||B||, city, IC = B0 and compression loss from a gauge-fixed trajectory."""
    if mask is not None:
        S = S[mask]
    S = S[np.linalg.norm(S, axis=1) > 0]
    if not len(S):
        return {}
    B = S.mean(axis=0)
    nb = float(np.linalg.norm(B))
    city = B / nb if nb > 0 else B
    return {
        "concentration": nb,
        "IC_gauge": float(B[0]),
        "city0": float(city[0]),
        "compression_loss": float(B[0] * (1 - 1 / nb)) if nb > 0 else float("nan"),
        "city": city,
    }


def angle_deg(c1: np.ndarray, c2: np.ndarray) -> float:
    c = float(np.clip(np.dot(c1, c2), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))
