"""Run ANY research script of ANY cycle safely - old C4 archives included - without losing earlier outputs.

  python common/tools/run_research.py seasons/S1/C4/research/r06_official_screen.py
  python common/tools/run_research.py seasons/S1/C4/research/r16_x_alpha_screen.py
  python common/tools/run_research.py common/research_templates/r00_data_audit.py      (runs on the ACTIVE cycle)
  python common/tools/run_research.py <script.py> -- <script args...>

What it guarantees:
  * cycle context: a script under seasons/<S>/<C>/ runs with ALPHANOVA_CYCLE=<S>/<C> (whatever ACTIVE_CYCLE says), its
    working directory = that cycle folder (so the old root-relative paths 'research/...', 'strategies/...',
    'reports/...' resolve exactly as when the script was written) and `lab` importable (common/ on PYTHONPATH);
  * nothing is lost: every file in the cycle's reports/, config/ and research/ (except large caches) is
    snapshotted first; any file the script overwrites or removes keeps its previous version in <dir>/_history/;
  * every run is logged to <cycle>/experiments/research_runs.csv (utc, script, args, exit code, files changed).
"""
from __future__ import annotations

import csv
import hashlib
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "common"
WATCH = ("reports", "config", "research")
MAX_BYTES = 20_000_000          # larger files (caches) are not snapshotted
SKIP_PARTS = {"_history", "__pycache__", "_offline_learned"}


def cycle_of(script: Path) -> tuple[str, Path]:
    parts = script.resolve().parts
    if "seasons" in parts:
        i = parts.index("seasons")
        if len(parts) > i + 2:
            sid, cid = parts[i + 1], parts[i + 2]
            return f"{sid}/{cid}", ROOT / "seasons" / sid / cid
    cur = os.environ.get("ALPHANOVA_CYCLE") or (ROOT / "ACTIVE_CYCLE").read_text(encoding="utf-8").strip()
    sid, cid = cur.split("/", 1)
    return cur, ROOT / "seasons" / sid / cid


def files(cyc: Path):
    for d in WATCH:
        base = cyc / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and not (SKIP_PARTS & set(p.relative_to(cyc).parts)) and p.stat().st_size <= MAX_BYTES:
                yield p


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 1
    script = Path(argv[0]).resolve()
    extra = argv[1:]
    if extra and extra[0] == "--":
        extra = extra[1:]
    cid, cyc = cycle_of(script)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    snap = cyc / "experiments" / f"_snapshot_tmp_{stamp}"
    before = {}
    for p in files(cyc):
        rel = p.relative_to(cyc)
        before[rel] = sha(p)
        dst = snap / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    env = dict(os.environ, ALPHANOVA_CYCLE=cid, PYTHONIOENCODING="utf-8", COMP6_DISABLE_PIP_INSTALL="1",
               PYTHONPATH=os.pathsep.join([str(COMMON), str(script.parent), os.environ.get("PYTHONPATH", "")]))
    print(f"[run_research] cycle {cid} | cwd {cyc} | {script.name} {' '.join(extra)}", flush=True)
    rc = subprocess.call([sys.executable, str(script), *extra], cwd=str(cyc), env=env)
    changed = []
    for rel, h in before.items():
        cur = cyc / rel
        if not cur.exists() or sha(cur) != h:
            hist = (cyc / rel).parent / "_history"
            hist.mkdir(parents=True, exist_ok=True)
            keep = hist / f"{rel.stem}.before_{stamp}{rel.suffix}"
            shutil.copy2(snap / rel, keep)
            changed.append(f"{rel.as_posix()} (previous kept: {keep.relative_to(cyc).as_posix()})")
    shutil.rmtree(snap, ignore_errors=True)   # removes only the temporary copies made above
    log = cyc / "experiments" / "research_runs.csv"
    new = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc", "script", "args", "exit_code", "files_changed"])
        w.writerow([stamp, script.relative_to(ROOT).as_posix() if script.is_relative_to(ROOT) else str(script),
                    " ".join(extra), rc, " | ".join(changed)])
    print(f"[run_research] exit {rc}; {len(changed)} earlier file version(s) preserved in _history/")
    for c in changed:
        print("   ", c)
    return rc


if __name__ == "__main__":
    sys.exit(main())
