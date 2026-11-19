"""Shared bootstrapping for research scripts (works from ANY folder: finds <ROOT>/common by walking up)."""
from __future__ import annotations

import sys
from pathlib import Path


def _find_common() -> Path:
    for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (p / "common" / "lab" / "paths.py").exists():
            return p / "common"
    raise RuntimeError("cannot locate <ROOT>/common")


COMMON = _find_common()
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from lab.io import dev_vault_split, load_period  # noqa: E402
from lab.paths import CYCLE_ID, REPORTS, data_root, is_synthetic, settings  # noqa: E402


def setup(data: str | None = None):
    """Load settings and the latest panel of the ACTIVE cycle restricted to DEV rows (the vault is never looked at here)."""
    s = settings()
    root = data_root(data)
    panel = load_period(root)
    dev, _, dev_months, vault_months = dev_vault_split(panel, int(s["vault_months"]))
    tag = "synthetic" if is_synthetic(root) else "real"
    REPORTS.mkdir(parents=True, exist_ok=True)
    print(f"[{CYCLE_ID} | {tag}] period {panel.period}: {len(panel.index)} rows, dev months {dev_months[0]}..{dev_months[-1]}, "
          f"vault {vault_months} (excluded)")
    if tag == "synthetic":
        print("  NOTE: synthetic panel -> validates the code path only.")
    return s, panel.rows(dev), tag


def md_table(df, floatfmt="{:+.4f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r.tolist()) + " |")
    return "\n".join(out)
