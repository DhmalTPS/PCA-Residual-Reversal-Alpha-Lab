"""Write the synthetic test panel to data_synth/train/ (tool testing only -- never for fitting submissions)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.paths import SYNTHETIC  # noqa: E402
from lab.synth import write_package  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=14)
    ap.add_argument("--periods", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = SYNTHETIC
    w = write_package(out, a.months, a.periods, seed=a.seed)
    print(f"wrote periods {w} to {out}")
