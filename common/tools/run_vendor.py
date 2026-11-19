"""Run the ACTIVE cycle's vendor runner.py on a submission file (from the vendor folder, with safe env).

  python common/tools/run_vendor.py seasons/S1/C4/submissions/uploaded/s09_pca_idio_reversal.py
  python common/tools/run_vendor.py <file> --full          # !! prints per-period Sharpe incl. VAULT periods
  python common/tools/run_vendor.py <file> --gauge-fix     # !! same warning: run only AFTER the vault decision

Sets PYTHONIOENCODING=utf-8 (Windows console) and COMP6_DISABLE_PIP_INSTALL=1 (no auto pip from submissions);
results are appended to <cycle>/experiments/runner_results.csv.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.paths import EXPERIMENTS, VENDOR  # noqa: E402


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 1
    sub = str(Path(argv[0]).resolve())
    env = dict(os.environ, PYTHONIOENCODING="utf-8", COMP6_DISABLE_PIP_INSTALL="1")
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "runner.py", sub, "--output", str(EXPERIMENTS / "runner_results.csv"), *argv[1:]]
    return subprocess.call(cmd, cwd=str(VENDOR), env=env)


if __name__ == "__main__":
    sys.exit(main())
