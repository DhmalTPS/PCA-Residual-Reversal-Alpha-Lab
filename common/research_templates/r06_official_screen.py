"""G3c - HONEST marginal screen under the official per-period protocol.

The single-file screen (r05) is optimistic: each period file is a separate reconstruction and the
history inside the latest file carries hindsight. Here every construction is computed on each dev
period file's OWN rows (warm-up included) and scored only on that file's last 800 rows - exactly
the rows a submission is scored on. No fitting is involved, so there is no leakage to purge.

Reported per construction (sign as built):
  IC     : mean per-hour IC vs the target over the 29 x 800 validation rows (NW t), BH q=10%
  stab   : share of periods whose mean IC has the overall sign
  SR_net : net Sharpe @5bp on returns after 168h finite smoothing (with in-file warm-up)

  python research/r06_official_screen.py   -> reports/r06_official_screen.md / .csv
"""
# %%
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from _common import REPORTS, md_table, setup
from lab.backtest import pnl, sharpe
from lab.metrics import benjamini_hochberg, ic_series, mean_se_nw, p_two_sided
from lab.official import periods_split

s, _panel, tag = setup()
root = _panel.root
dev_p, _ = periods_split(s, root)
lag = int(s["nw_lag_hours"])
HL, WIN, WARM = 168.0, 672, 800


def rank(df):
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).to_numpy()[:, None]
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))


def z(df):
    d = df.sub(df.mean(axis=1), axis=0)
    return d.div(d.std(axis=1).replace(0, np.nan), axis=0)


def smooth(S):
    """Causal finite exponential kernel (same weights as the strategies' _smooth), via an FIR filter."""
    from scipy.signal import lfilter
    w = 0.5 ** (np.arange(WIN) / HL)
    A = S.fillna(0.0).to_numpy(float)
    num = lfilter(w, [1.0], A, axis=0)
    den = lfilter(w, [1.0], np.ones((len(A), 1)), axis=0)
    return pd.DataFrame(num / den, index=S.index, columns=S.columns)


def constructions(F, names, tick):
    raw = {n: F[n].reindex(columns=tick).astype(float) for n in names}
    zz = {n: z(raw[n]) for n in names}
    c = {}
    for n in names:
        c[f"rank {n}"] = rank(raw[n])
        for k in (1, 6, 24, 120):
            c[f"rank d{k} {n}"] = rank(raw[n].diff(k))
    for a, b in itertools.combinations(names, 2):
        c[f"rank z{a}-z{b}"] = rank(zz[a] - zz[b])
        c[f"rank z{a}*z{b}"] = rank(zz[a] * zz[b])
    return c


acc_ic, acc_pnl, per_ic = {}, {}, {}
for p in dev_p:
    F = pd.read_parquet(root / f"{p}.small.features.parquet")
    T = pd.read_parquet(root / f"{p}.small.target.parquet")
    X = pd.read_parquet(root / f"{p}.small.returns.parquet")
    n = len(F)
    nv = min(800, int(0.2 * n))
    lo = n - nv - WARM - WIN
    F, T, X = F.iloc[lo:], T.iloc[lo:], X.iloc[lo:]
    tick = list(dict.fromkeys(F.columns.get_level_values(-1)))
    names = list(dict.fromkeys(F.columns.get_level_values(0)))
    vi = F.index[len(F) - nv:]
    for k, S in constructions(F, names, tick).items():
        S = S.fillna(0.0)
        rho = ic_series(S, T, 1).reindex(vi).dropna()
        acc_ic.setdefault(k, []).append(rho)
        per_ic.setdefault(k, []).append(rho.mean())
        Sm = smooth(S)
        acc_pnl.setdefault(k, []).append(pnl(Sm.loc[vi], X.loc[vi], 5.0)["net"])
    print(f"  [{p}] done", flush=True)

rows = []
for k in acc_ic:
    rho = pd.concat(acc_ic[k])
    ic, se = mean_se_nw(rho.to_numpy(), lag)
    pm = np.array(per_ic[k])
    rows.append({"construction": k, "IC": ic, "t": ic / se if se else np.nan, "p": p_two_sided(ic / se) if se else np.nan,
                 "stab": float((np.sign(pm) == np.sign(ic)).mean()), "SR_net_sm168": sharpe(pd.concat(acc_pnl[k]))})
df = pd.DataFrame(rows)
df["BH_q10"] = benjamini_hochberg(df["p"].fillna(1).to_numpy(), 0.10)
df = df.reindex(df["t"].abs().sort_values(ascending=False).index)
df.to_csv(REPORTS / "r06_official_screen.csv", index=False)
head = (f"{len(df)} constructions over {len(dev_p)} periods x 800 validation rows; "
        f"{int(df['BH_q10'].sum())} survive BH q=10%.")
(REPORTS / "r06_official_screen.md").write_text("\n".join([f"# R06 official-protocol screen ({tag})", "", head, "", md_table(df)]), encoding="utf-8")
print(head)
print(df.head(25).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
