"""G0 - data audit: shapes, warm-up zeros, gaps, missingness, feature behaviour, returns geometry.

  python research/r00_data_audit.py         -> reports/r00_data_audit.md
"""
# %%
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import REPORTS, md_table, setup
from lab.io import list_periods, load_period
from lab.metrics import row_cos

s, panel, tag = setup()
F, X, T, D = panel.features, panel.returns, panel.target, panel.dispersion
lines = [f"# R00 data audit ({tag}, period {panel.period})", ""]

# %% structure
idx = F.index
step = pd.Series(idx[1:] - idx[:len(idx) - 1]).value_counts().head(5) if isinstance(idx, pd.DatetimeIndex) else None
lines += ["## Structure",
          f"- rows {len(idx)}, tickers {len(panel.tickers)}, features {panel.feature_names}",
          f"- index {idx[0]} .. {idx[len(idx) - 1]}; most common steps: {dict(step) if step is not None else 'n/a'}",
          f"- periods on disk: {list_periods(panel.root)} (cumulative: only the latest is used)"]

zero_t = (T.abs().sum(axis=1) == 0)
first_nz = int(np.argmax(~zero_t.to_numpy())) if (~zero_t).any() else len(T)
lines += [f"- target warm-up: first {first_nz} rows all-zero; all-zero rows in total {int(zero_t.sum())}",
          f"- target range [{np.nanmin(T.to_numpy()):.2f}, {np.nanmax(T.to_numpy()):.2f}], clipped share {(T.abs() >= 4.999).to_numpy().mean():.4f}",
          f"- dispersion file: {'present' if D is not None else 'absent'} (FORWARD-looking: diagnostics only)"]

# %% rotation check across periods (expected: tickers re-labelled per period)
ps = list_periods(panel.root)
if len(ps) >= 2:
    a = load_period(panel.root, ps[len(ps) - 2]).returns
    b = load_period(panel.root, ps[len(ps) - 1]).returns.loc[a.index]
    same = np.mean([np.corrcoef(a[c].fillna(0), b[c].fillna(0))[0, 1] for c in a.columns if c in b.columns])
    lines.append(f"- mean corr of same-named ticker across the last two periods: {same:+.3f} (~1 would mean NO rotation)")

# %% missingness / features
rows = []
for f in panel.feature_names:
    B = F[f]
    d = B.diff()
    ac1 = np.nanmean([B[c].autocorr(1) for c in B.columns])
    dac1 = np.nanmean([d[c].autocorr(1) for c in d.columns])
    xs = B.sub(B.mean(axis=1), axis=0)
    link = row_cos(d, X).mean()
    rows.append({"feature": f, "nan_frac": float(B.isna().to_numpy().mean()), "level_ac1": float(ac1), "diff_ac1": float(dac1),
                 "xs_sd_median": float(xs.std(axis=1).median()), "scale_ratio_max_min": float(B.std().max() / max(B.std().min(), 1e-12)),
                 "cos(dF, X)": float(link)})
fdf = pd.DataFrame(rows)
lines += ["", "## Features", "level_ac1 near 1 = slow state; cos(dF, X) far from 0 = feature changes co-move with returns (return-like).", md_table(fdf)]

# %% returns geometry -> effective breadth
Xd = X.sub(X.mean(axis=1), axis=0).dropna()
C = np.corrcoef(Xd.to_numpy().T)
J = C.shape[0]
rho_bar = (C.sum() - J) / (J * (J - 1))
ev = np.sort(np.linalg.eigvalsh(C))[::-1]
n_eff = (ev.sum() ** 2) / (ev ** 2).sum()
lines += ["", "## Returns geometry",
          f"- hourly sd (median asset) {X.std().median():.5f}; mean |X| {X.abs().to_numpy().mean():.5f}",
          f"- mean pairwise corr of de-meaned returns {rho_bar:+.3f}; participation-ratio breadth N_eff = {n_eff:.1f} of {J}",
          f"- top eigenvalue share {ev[0] / ev.sum():.3f}",
          f"- fundamental-law scale: IC 0.01 x sqrt(N_eff) = gross hourly SR ~ {0.01 * np.sqrt(n_eff):.3f}"]

out = REPORTS / "r00_data_audit.md"
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print(f"\n-> {out}")
