"""City / concentration / compression loss of a run, on DEV periods only (keeps the vault clean).

  python tools/city.py RUN_ID [--top 10]

Uses the vendor's city_tools on the stored out-of-sample predictions of an official-protocol run
(experiments/runs/RUN_ID/preds.parquet, index "NNN:row"): for each period, the shipped gauge
matrices (800, 19, 20) are applied to the predictions in numeric ticker order, then the city is
compared with signal_db (1646 known cities). Unlike `runner.py --gauge-fix` this never touches
the vault periods. Remember: city novelty is only a proxy; admission is |rho| <= 0.5 (global).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lab.paths import REPORTS, RUNS, VENDOR, archive_existing, data_root  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id")
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(VENDOR))
    import city_tools  # vendor module of the active cycle

    P = pd.read_parquet(RUNS / a.run_id / "preds.parquet")
    per = pd.Series(P.index).str.split(":").str[0].to_numpy()
    root = data_root()
    gfs = []
    for p in dict.fromkeys(per):
        G = np.load(root / "gauge" / f"{p}.small.gauge.npy")
        Pp = P[per == p]
        cols = sorted(Pp.columns, key=lambda c: int(str(c).split(".")[-1]))
        A = Pp[cols].to_numpy(float)
        A = A - A.mean(axis=1, keepdims=True)
        n = min(len(A), G.shape[0])
        gfs.append(city_tools.compute_gauge_fixed(A[len(A) - n:], G[G.shape[0] - n:]))
    gf = np.concatenate(gfs)
    ic = city_tools.compute_ic(gf)
    city = city_tools.compute_city(gf)
    cmp = city_tools.compare_to_cities(city, str(VENDOR / "signal_db" / "signal_cities_ambient_coords.parquet"))
    out = {"run_id": a.run_id, "periods": len(gfs), "rows": int(len(gf)), "IC_gauge": ic["ic_mean"], "IC_std": ic["ic_std"],
           "concentration": ic["concentration"], "compression_loss": ic["ic_mean"] - float(city[0]),
           "city_novelty_deg": float(cmp.iloc[0]["angular_distance_deg"]), "nearest": str(cmp.iloc[0]["signal_id"]),
           "n_within_60deg": int((cmp["angular_distance_deg"] < 60).sum()),
           "nearest_list": cmp.head(a.top).to_dict("records")}
    print(f"run {a.run_id}: {out['periods']} dev periods, {out['rows']} rows")
    print(f"  IC (gauge e0) {out['IC_gauge']:+.4f}  IC std {out['IC_std']:.4f}  concentration {out['concentration']:.4f}  "
          f"compression loss {out['compression_loss']:+.4f}")
    print(f"  city novelty {out['city_novelty_deg']:.1f} deg (nearest {out['nearest']}); cities within 60 deg: {out['n_within_60deg']} of {len(cmp)}")
    print(cmp.head(a.top).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    REPORTS.mkdir(exist_ok=True)
    archive_existing(REPORTS / f"city_{a.run_id}.json").write_text(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
