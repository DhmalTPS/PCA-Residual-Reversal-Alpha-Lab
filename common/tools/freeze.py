"""Promote a candidate to submit/ (before upload) or freeze it into pot/ (after the server accepts it).

  python tools/freeze.py promote strategies/s02_lgbm_regime.py --set SMOOTH_HALFLIFE=12
      -> writes submit/<name>.py with the overrides BAKED IN as class attributes, re-runs the
         static + dynamic gates on that exact file, and writes submit/<name>.checklist.md.
  python tools/freeze.py pot submit/<name>.py --cycle C4 --verdict accepted
      -> copies the file into pot/<cycle>/<name>/, re-generates its predictions on the
         current data and stores them, so every later candidate is checked against it.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.io import load_period  # noqa: E402
from lab.loader import file_sha, load_strategy, parse_overrides  # noqa: E402
from lab.paths import COMMON, CYCLE, POT, SUBMIT, archive_existing, data_root, settings  # noqa: E402

PY = sys.executable


def bake(src: Path, overrides: dict) -> str:
    text = src.read_text(encoding="utf-8")
    for k, v in overrides.items():
        pat = re.compile(rf"^(    {re.escape(k)} = ).*$", re.MULTILINE)
        if not pat.search(text):
            raise SystemExit(f"cannot bake {k}: no class-level assignment found")
        text = pat.sub(lambda m: m.group(1) + repr(v), text, count=1)
    return text


CHECKLIST = """# Submission checklist - {name}

Source: `{src}`  overrides baked: `{ov}`  sha: `{sha}`  created: {utc}

Automated (re-run on this exact file):
- [{static}] static gate (tools/precheck.py)
- [{dynamic}] runtime gate (tools/dynamic_check.py)

Evidence (fill from experiments/registry.csv and reports/):
- [ ] full walk-forward: NW 90% CI of net Sharpe @10bp excludes 0
- [ ] deflated Sharpe prob > 0.5 with the honest trial count
- [ ] shuffle gate p < 0.05 (tools/shuffle_gate.py)
- [ ] |rho| 90% upper bound <= 0.35 vs pot/ and vs the other submissions of this cycle (tools/choose.py)
- [ ] `python runner.py {name}.py` (official quick + de-mean check) PASS
- [ ] `python runner.py {name}.py --gauge-fix`: city distance recorded
- [ ] vault run once, same sign as dev, inside the dev CI (tools/vault.py)
- [ ] submitted >= 24 h before the deadline

Rules compliance:
- [ ] own original work; only OSI-licensed libraries (numpy/pandas/scikit-learn/lightgbm: BSD/MIT)
- [ ] no external data; no hand labelling
- [ ] code not shared privately with anyone
"""


def promote(src: Path, overrides: dict) -> int:
    SUBMIT.mkdir(exist_ok=True)
    tag = "".join(f"_{k.lower()}{v}" for k, v in overrides.items()).replace(".", "p").replace(" ", "")[:60]
    name = f"{src.stem}{tag}"
    dst = SUBMIT / f"{name}.py"
    archive_existing(dst).write_text(bake(src, overrides), encoding="utf-8")
    st = subprocess.run([PY, str(COMMON / "tools" / "precheck.py"), str(dst)], capture_output=True, text=True)
    dy = subprocess.run([PY, str(COMMON / "tools" / "dynamic_check.py"), str(dst)], capture_output=True, text=True)
    print(st.stdout.strip())
    print(dy.stdout.strip())
    archive_existing(SUBMIT / f"{name}.checklist.md").write_text(CHECKLIST.format(
        name=name, src=src.relative_to(CYCLE).as_posix() if src.is_relative_to(CYCLE) else src, ov=json.dumps(overrides),
        sha=file_sha(dst), utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        static="x" if st.returncode == 0 else " ", dynamic="x" if dy.returncode == 0 else " "))
    ok = st.returncode == 0 and dy.returncode == 0
    print(f"\n{'READY' if ok else 'NOT READY'}: {dst}")
    return 0 if ok else 1


def to_pot(src: Path, cycle: str, verdict: str) -> int:
    d = POT / cycle / src.stem
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, d / src.name)
    s = settings()
    panel = load_period(data_root())
    cls = load_strategy(d / src.name)
    m = cls()
    m.train(panel.features, panel.target)
    P = m.predict(panel.features)
    P.to_parquet(d / "preds.parquet")
    (d / "meta.json").write_text(json.dumps({"cycle": cycle, "verdict": verdict, "sha": file_sha(src), "period": panel.period,
                                             "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                             "note": "in-sample predictions on the full panel; used only for rho_hat vs new candidates"}, indent=2))
    print(f"frozen into {d}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("promote")
    p1.add_argument("file")
    p1.add_argument("--set", action="append", default=[])
    p2 = sub.add_parser("pot")
    p2.add_argument("file")
    p2.add_argument("--cycle", required=True)
    p2.add_argument("--verdict", default="accepted")
    a = ap.parse_args(argv)
    if a.cmd == "promote":
        return promote(Path(a.file).resolve(), parse_overrides(a.set))
    return to_pot(Path(a.file).resolve(), a.cycle, a.verdict)


if __name__ == "__main__":
    sys.exit(main())
