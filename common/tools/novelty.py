"""Novelty of candidate runs vs each other and vs YOUR legacy pot (pot/*/preds.parquet).

  python tools/novelty.py                      # latest full run of every strategy
  python tools/novelty.py --runs RUN_ID RUN_ID

The admission rule is |rho_hat| <= 0.5 against every higher-ranked signal and the pot
(verified on the public board: Q4 admitted at rho 0.490). We require the bootstrap 90% upper
bound of |rho_hat| <= novelty_rho_max (0.35) because the score is computed on a longer history
(hidden band + live) than ours. Other participants' signals are invisible locally: for those,
use `runner.py --gauge-fix` (city distance) and the board's global-novelty column after submitting.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from lab import registry  # noqa: E402
from lab.gauge import rho_hat  # noqa: E402
from lab.paths import POT, REPORTS, RUNS, archive_existing, settings  # noqa: E402


def load_signals(run_ids: list[str] | None, data_tag: str | None) -> dict[str, pd.DataFrame]:
    sig = {}
    if run_ids:
        for r in run_ids:
            sig[r] = pd.read_parquet(RUNS / r / "preds.parquet")
    else:
        for _, row in registry.latest_runs("full", data_tag).iterrows():
            label = row["strategy"] + (f"{row['overrides']}" if row["overrides"] else "")
            sig[label] = pd.read_parquet(RUNS / row["run_id"] / "preds.parquet")
    return sig


def load_pot() -> dict[str, pd.DataFrame]:
    out = {}
    for p in sorted(POT.glob("*/*/preds.parquet")):
        out[f"POT:{p.parent.parent.name}/{p.parent.name}"] = pd.read_parquet(p)
    return out


def matrix(signals: dict, s: dict) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(signals, 2):
        r = rho_hat(signals[a], signals[b], float(s["bootstrap_block_hours"]), 500, int(s["seed"]))
        rows.append({"a": a, "b": b, **r})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--data-tag", default=None, help="'real' or 'synthetic' (default: all)")
    a = ap.parse_args(argv)
    s = settings()
    sig = load_signals(a.runs, a.data_tag)
    pot = load_pot()
    allsig = {**sig, **pot}
    if len(allsig) < 2:
        print("need at least two signals (runs and/or pot entries)")
        return 1
    df = matrix(allsig, s)
    thr = float(s["novelty_rho_max"])
    df["status"] = df["abs_ub"].map(lambda v: "OK" if v <= thr else ("RISK" if v <= float(s["official_rho_max"]) else "BLOCK"))
    pd.set_option("display.width", 200)
    print(df[["a", "b", "rho", "lo", "hi", "abs_ub", "hours", "status"]].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(f"\nOK: |rho| 90% upper bound <= {thr}; RISK: <= 0.5 (admissible today, fragile on the hidden band); BLOCK: > 0.5")
    REPORTS.mkdir(exist_ok=True)
    df.to_csv(archive_existing(REPORTS / "novelty_matrix.csv"), index=False)
    archive_existing(REPORTS / "novelty_matrix.json").write_text(json.dumps(df.to_dict("records"), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
