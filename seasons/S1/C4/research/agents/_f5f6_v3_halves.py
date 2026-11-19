# Diagnostic only (NOT a new trial): per-period net SR of already-evaluated constructions, split 001-014 / 015-029.
import sys; sys.path.insert(0, ".")
import importlib.util, numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location("v3", "research/agents/F5F6_NONLINEAR_v3.py"); v3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3)
from lab.honest import _load, dev_periods, unit, smooth, VAL
from lab.backtest import pnl, sharpe
for nm, fn in (("combo_e1e2", v3.fn_combo_e1e2), ("crev_f5act", v3.fn_crev_f5act), ("crev_f6ext", v3.fn_crev_f6ext)):
    per = []
    for p in dev_periods():
        D = _load(p, VAL + 4 * 672 + 1400); F, X = D["features"], D["returns"]
        vi = F.index[len(F) - VAL:]
        S = smooth(unit(fn(F).reindex(index=F.index, columns=X.columns)), 672.0).loc[vi]
        per.append(pnl(S, X.loc[vi], 5.0)["net"].to_numpy())
    h1, h2 = np.concatenate(per[:14]), np.concatenate(per[14:])
    print(nm, "H1", round(sharpe(h1), 4), "H2", round(sharpe(h2), 4), "per-period", np.round([sharpe(x) for x in per], 3).tolist())
