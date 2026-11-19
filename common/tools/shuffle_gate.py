"""Local overfitting-gate emulation (shortlisted candidates only: it retrains reps+1 times).

  python tools/shuffle_gate.py strategies/s04_xs_contemp.py --reps 30
  python tools/shuffle_gate.py demo_engineered.py --reps 30          # calibration: vendor says p~0.01

Reference slice = the TRAINING part of the latest development period (all dev rows except the last
800, like the official split), optionally capped to the last --rows rows.
PASS if p(net Sharpe) < alpha. Both p-values (Sharpe on returns, IC on target) are reported.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from lab.official import periods_split  # noqa: E402
from lab.loader import load_strategy  # noqa: E402
from lab.nulls import shuffle_gate  # noqa: E402
from lab.paths import REPORTS, archive_existing, data_root, settings  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--rows", type=int, default=None, help="cap the reference slice to its last N rows")
    ap.add_argument("--mode", choices=["server", "label_block"], default="server")
    ap.add_argument("--data", default=None)
    a = ap.parse_args(argv)
    os.environ.setdefault("COMP6_DISABLE_PIP_INSTALL", "1")
    s = settings()
    reps = a.reps or int(s["shuffle_gate_reps"])
    h = int(s["target_horizon_hours"])
    root = data_root(a.data)
    dev_p, _ = periods_split(s, root)
    ref = dev_p[-1]
    F, T, X = (pd.read_parquet(root / f"{ref}.small.{k}.parquet") for k in ("features", "target", "returns"))
    n = len(F) - 800
    lo = max(0, n - a.rows) if a.rows else 0
    F, T, X = F.iloc[lo:n], T.iloc[lo:n], X.iloc[lo:n]
    print(f"reference = training slice of period file {ref}")
    cls = load_strategy(a.file)
    print(f"shuffle_gate {Path(a.file).stem}: mode={a.mode} reps={reps} h={h} rows={len(F)}")
    t0 = time.perf_counter()
    r = shuffle_gate(cls, F, T, X, int(s["target_lag"]), h, reps, int(s["seed"]), float(s["official_cost_bp"]), a.mode, h)
    r["seconds"] = round(time.perf_counter() - t0, 1)
    r["rows"] = len(F)
    r["passed"] = r["p_sharpe"] < float(s["shuffle_gate_alpha"])
    print(f"  Sharpe: real {r['real_sharpe']:+.5f} | null {r['null_sharpe_mean']:+.5f} +- {r['null_sharpe_sd']:.5f} (q95 {r['null_sharpe_q95']:+.5f}) "
          f"z {r['z_sharpe']:.2f} p {r['p_sharpe']:.3f}")
    print(f"  IC    : real {r['real_ic']:+.5f} | null {r['null_ic_mean']:+.5f} +- {r['null_ic_sd']:.5f} (q95 {r['null_ic_q95']:+.5f}) "
          f"z {r['z_ic']:.2f} p {r['p_ic']:.3f}")
    print(f"  -> {'PASS' if r['passed'] else 'FAIL'} (alpha {s['shuffle_gate_alpha']}, on Sharpe)  [{r['seconds']} s]")
    REPORTS.mkdir(exist_ok=True)
    archive_existing(REPORTS / f"shuffle_{Path(a.file).stem}.json").write_text(json.dumps(r, indent=2))
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
