"""Self-test: every gate must PASS the clean strategies and CATCH each deliberately bad control.

  python tests/selftest.py            (uses data/train if present, else data_synth/train)

A gate that cannot catch its own control cannot be trusted on a real submission.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

COMMON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMMON))
from lab.paths import STRATEGIES  # noqa: E402  (active cycle's strategies)

# clean reference for the runtime gate: a proven common template (independent of any cycle's legacy files)
REFERENCE = COMMON / "strategy_templates" / "t_factor_residual_reversal__from_S1C4_s09.py"

ROOT = COMMON
PY = sys.executable
CTRL = COMMON / "tests" / "controls"


def rc(args) -> tuple[int, str]:
    p = subprocess.run([PY, *map(str, args)], capture_output=True, text=True, cwd=ROOT)
    return p.returncode, p.stdout + p.stderr


def expect(name, got_rc, want_pass, out, must_contain=None):
    ok = (got_rc == 0) == want_pass and (must_contain is None or must_contain in out)
    print(f"  [{'ok' if ok else 'XX'}] {name}: expected {'PASS' if want_pass else 'FAIL'}"
          f"{' containing ' + repr(must_contain) if must_contain else ''}, got rc={got_rc}")
    if not ok:
        print("      " + "\n      ".join(out.strip().splitlines()[-12:]))
    return ok


def main() -> int:
    results = []
    print("static gate (precheck)")
    for f in sorted(STRATEGIES.glob("s*.py")) + sorted((COMMON / "strategy_templates").glob("t_*.py")):
        r, o = rc([ROOT / "tools" / "precheck.py", f])
        results.append(expect(f.name, r, True, o))
    for f, want, needle in (("c00_noise.py", True, None), ("c01_leaky.py", False, "negative shift"),
                            ("c02_rowmask.py", True, None), ("c03_toplevel.py", False, "CODE_OUTSIDE_CLASS")):
        r, o = rc([ROOT / "tools" / "precheck.py", CTRL / f])
        results.append(expect(f, r, want, o, needle))

    print("runtime gate (dynamic_check)")
    r, o = rc([ROOT / "tools" / "dynamic_check.py", REFERENCE])
    results.append(expect(REFERENCE.name, r, True, o))
    r, o = rc([ROOT / "tools" / "dynamic_check.py", CTRL / "c01_leaky.py"])
    results.append(expect("c01_leaky.py", r, False, o, "[FAIL] causality"))
    r, o = rc([ROOT / "tools" / "dynamic_check.py", CTRL / "c02_rowmask.py"])
    results.append(expect("c02_rowmask.py", r, False, o, "[FAIL] row_shift"))

    print("overfitting gate (shuffle_gate, 8 reps)")
    r, o = rc([ROOT / "tools" / "shuffle_gate.py", CTRL / "c00_noise.py", "--reps", "8"])
    results.append(expect("c00_noise.py", r, False, o))

    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} expectations met -> {'SELFTEST PASS' if n_ok == len(results) else 'SELFTEST FAIL'}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
