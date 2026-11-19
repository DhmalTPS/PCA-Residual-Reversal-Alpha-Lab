"""Show or switch the active cycle (written to <ROOT>/ACTIVE_CYCLE).

  python common/tools/set_cycle.py            # show the active cycle and list all cycles
  python common/tools/set_cycle.py S1/C5      # switch (the cycle folder must exist: see new_cycle.py)

Every tool and research script works on the active cycle. For a one-off command on another cycle,
set the environment variable ALPHANOVA_CYCLE instead (PowerShell: $env:ALPHANOVA_CYCLE="S1/C4").
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEASONS = ROOT / "seasons"


def cycles() -> list[str]:
    out = []
    for s in sorted(p for p in SEASONS.glob("S*") if p.is_dir()):
        for c in sorted(p for p in s.glob("C*") if p.is_dir()):
            out.append(f"{s.name}/{c.name}")
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    f = ROOT / "ACTIVE_CYCLE"
    cur = f.read_text(encoding="utf-8").strip() if f.exists() else "(none)"
    if not argv:
        print(f"active cycle: {cur}")
        for c in cycles():
            print(f"  {'*' if c == cur else ' '} {c}")
        return 0
    want = argv[0].strip().strip("/")
    if want not in cycles():
        print(f"cycle {want} does not exist. Create it first: python common/tools/new_cycle.py {want} --zip <package.zip>")
        return 1
    f.write_text(want + "\n", encoding="utf-8")
    print(f"active cycle: {cur} -> {want}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
