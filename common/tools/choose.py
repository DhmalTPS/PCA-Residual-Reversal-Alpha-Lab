"""Choose WHICH candidates to submit: an emulation of the official greedy quality-set rule,
run on your own candidates + your pot, scored on the shrunk (posterior) Sharpe.

  python tools/choose.py --data-tag real

Order: posterior-mean Sharpe (net, official cost), descending. A candidate is admitted if
  * P(true SR > 0) >= min_prob_positive,
  * net Sharpe at the STRESS cost is > 0 (robust to the extra official cost adjustments),
  * its |rho_hat| 90% upper bound <= novelty_rho_max vs every pot entry and every admitted one.
Rejected candidates are listed with the reason. Submitting near-duplicates is pointless (only the
best counts) and harmful (every passing submission enters the legacy pot for the rest of the season).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lab import registry  # noqa: E402
from lab.gauge import rho_hat  # noqa: E402
from lab.paths import CONFIG, REPORTS, RUNS, archive_existing, settings  # noqa: E402
from tools.novelty import load_pot  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-tag", default="real")
    ap.add_argument("--max", type=int, default=3, help="max submissions to recommend this cycle")
    a = ap.parse_args(argv)
    s = settings()
    reg = registry.latest_runs("full", a.data_tag)
    if not len(reg):
        print(f"no full runs for data={a.data_tag}")
        return 1
    reg = reg.copy()
    reg["SR_shrunk"] = pd.to_numeric(reg["SR_shrunk"], errors="coerce")
    reg = reg.sort_values("SR_shrunk", ascending=False)
    pot = load_pot()
    thr = float(s["novelty_rho_max"])
    lb = json.loads((CONFIG / "leaderboard.json").read_text()) if (CONFIG / "leaderboard.json").exists() else {}

    admitted, report = [], []
    for _, row in reg.iterrows():
        label = row["strategy"] + (row["overrides"] or "")
        P = pd.read_parquet(RUNS / row["run_id"] / "preds.parquet")
        reasons = []
        if not (float(row["P_SR_pos"]) >= float(s["min_prob_positive"])):
            reasons.append(f"P(SR>0)={float(row['P_SR_pos']):.2f} < {s['min_prob_positive']}")
        if not (float(row["SR_net_10bp"]) > 0):
            reasons.append(f"stress SR@10bp={float(row['SR_net_10bp']):+.4f} <= 0")
        worst = ("", 0.0)
        for name, Q in list(pot.items()) + [(x["label"], x["P"]) for x in admitted]:
            r = rho_hat(P, Q, float(s["bootstrap_block_hours"]), 300, int(s["seed"]))
            if np.isfinite(r["abs_ub"]) and r["abs_ub"] > worst[1]:
                worst = (name, r["abs_ub"])
        if worst[1] > thr:
            reasons.append(f"|rho| ub {worst[1]:.2f} vs {worst[0]}")
        ok = not reasons and len(admitted) < a.max
        if ok:
            admitted.append({"label": label, "P": P, "run_id": row["run_id"]})
        report.append({"candidate": label, "run_id": row["run_id"], "SR_net_5bp": float(row["SR_net_5bp"]),
                       "SR_shrunk": float(row["SR_shrunk"]), "P_SR_pos": float(row["P_SR_pos"]),
                       "SR_net_10bp": float(row["SR_net_10bp"]), "DSR": float(row["DSR"]),
                       "max_rho_ub": worst[1], "vs": worst[0], "decision": "SUBMIT" if ok else "hold",
                       "why": "; ".join(reasons) or ("cap reached" if not ok else "")})
    df = pd.DataFrame(report)
    pd.set_option("display.width", 220)
    print(df.drop(columns=["run_id"]).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    if lb:
        print(f"\nboard bar ({lb['as_of_utc'][:10]}): 3rd {lb['sharpe_third']}, 2nd {lb['sharpe_second']}, 1st {lb['sharpe_first']} (hourly, net)")
    REPORTS.mkdir(exist_ok=True)
    df.to_csv(archive_existing(REPORTS / f"selection_{a.data_tag}.csv"), index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
