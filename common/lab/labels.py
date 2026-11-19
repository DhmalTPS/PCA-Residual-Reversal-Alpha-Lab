"""LOOK-AHEAD ZONE. The only module allowed to look past hour t.

Used exclusively for offline identification of the target (horizon, alignment,
residualisation). Nothing here may ever be copied into strategies/ or submit/ --
tools/precheck.py rejects any strategy that imports `lab`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lab.metrics import row_cos


def window_sum(X: pd.DataFrame, h: int, offset: int) -> pd.DataFrame:
    """sum_{k=offset}^{offset+h-1} X(i+k)  (look-ahead by construction)."""
    acc = None
    for k in range(offset, offset + h):
        s = X.shift(-k)
        acc = s if acc is None else acc + s
    return acc


def xs_std(df: pd.DataFrame) -> pd.Series:
    return df.sub(df.mean(axis=1), axis=0).std(axis=1, ddof=0)


def horizon_scan(X: pd.DataFrame, T: pd.DataFrame, dispersion: pd.Series | None,
                 horizons=(1, 4, 12, 24, 48, 72, 96, 108, 120, 132, 144, 168, 240), offsets=(0, 1, 2)) -> pd.DataFrame:
    """For each (offset, h): mean per-hour cosine between target and the forward window sum,
    and (if dispersion ships) the correlation / ratio between sigma(i) and the xs std of that sum."""
    rows = []
    valid = T.std(axis=1) > 1e-8
    for off in offsets:
        for h in horizons:
            R = window_sum(X, h, off)
            c = row_cos(R.loc[valid], T.loc[valid]).dropna()
            row = {"offset": off, "h": h, "cos_mean": float(c.mean()), "cos_median": float(c.median()), "hours": len(c)}
            if dispersion is not None:
                sd = xs_std(R).reindex(dispersion.index)
                ok = sd.notna() & dispersion.notna() & (dispersion > 0) & valid.reindex(dispersion.index, fill_value=False)
                if ok.sum() > 10:
                    row["disp_corr"] = float(np.corrcoef(sd[ok], dispersion[ok])[0, 1])
                    row["disp_ratio_median"] = float((dispersion[ok] / sd[ok].replace(0, np.nan)).median())
            rows.append(row)
    return pd.DataFrame(rows)


def target_autocorr(T: pd.DataFrame, lags=(1, 2, 4, 8, 12, 24, 48, 96)) -> dict:
    """mean_i cos(T(i), T(i+k)): how fast the target itself decorrelates (upper bound on rho_alpha)."""
    valid = T.std(axis=1) > 1e-8
    Tv = T.where(valid, np.nan)
    return {int(k): float(row_cos(Tv, Tv.shift(-k)).mean()) for k in lags}


def residual_audit(X: pd.DataFrame, T: pd.DataFrame, dispersion: pd.Series, h: int, offset: int,
                   regressors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Reconstruct R~ = sigma * T, compare with the window sum, and check which simple
    cross-sectional regressors explain the gap (i.e. what the target was neutralised against)."""
    R = window_sum(X, h, offset)
    Rd = R.sub(R.mean(axis=1), axis=0)
    Rt = T.mul(dispersion, axis=0)
    gap = Rt - Rd
    out = [{"name": "_fit_cos(R~, windowsum)", "cos_with_gap": float(row_cos(Rt, Rd).mean())}]
    for name, reg in regressors.items():
        out.append({
            "name": name,
            "cos_with_gap": float(row_cos(gap, reg).mean()),
            "cos_with_windowsum": float(row_cos(Rd, reg).mean()),
            "cos_with_target": float(row_cos(T, reg).mean()),
        })
    return pd.DataFrame(out)
