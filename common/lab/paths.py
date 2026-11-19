"""Project paths - CYCLE-AWARE.

Layout:
    <ROOT>/common/...                       reusable code shared by every cycle (this package lives in common/lab)
    <ROOT>/seasons/<SEASON>/pot/            our own legacy-pot copies for that season
    <ROOT>/seasons/<SEASON>/<CYCLE>/...     everything specific to one cycle (vendor package, data, config,
                                            strategies, research, experiments, reports, submissions, decisions)

The active cycle comes from the environment variable ALPHANOVA_CYCLE (e.g. "S1/C5") if set, else from the
file <ROOT>/ACTIVE_CYCLE. Switch cycles with:  python common/tools/set_cycle.py S1/C5
Settings = common/config/defaults.json, overlaid by <cycle>/config/settings.json, then <cycle>/config/derived.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
COMMON = LAB.parent
ROOT = COMMON.parent
SEASONS = ROOT / "seasons"
STUB = LAB / "_stub"
SYNTHETIC = COMMON / "synthetic" / "train"


def active_cycle() -> str:
    v = os.environ.get("ALPHANOVA_CYCLE")
    if not v:
        f = ROOT / "ACTIVE_CYCLE"
        v = f.read_text(encoding="utf-8").strip() if f.exists() else ""
    if not v or "/" not in v:
        raise RuntimeError("No active cycle. Run: python common/tools/set_cycle.py S1/C4  (or set ALPHANOVA_CYCLE)")
    return v.strip().strip("/")


CYCLE_ID = active_cycle()
SEASON_ID, CYCLE_NAME = CYCLE_ID.split("/", 1)
SEASON = SEASONS / SEASON_ID
CYCLE = SEASON / CYCLE_NAME

VENDOR = CYCLE / "vendor"
CONFIG = CYCLE / "config"
EXPERIMENTS = CYCLE / "experiments"
RUNS = EXPERIMENTS / "runs"
REGISTRY = EXPERIMENTS / "registry.csv"
VAULT_LOG = EXPERIMENTS / "vault_log.csv"
REPORTS = CYCLE / "reports"
STRATEGIES = CYCLE / "strategies"
SUBMIT = CYCLE / "submissions"
RESEARCH = CYCLE / "research"
POT = SEASON / "pot"
COMMON_CONFIG = COMMON / "config"


def settings() -> dict:
    s = json.loads((COMMON_CONFIG / "defaults.json").read_text())
    for f in (CONFIG / "settings.json", CONFIG / "derived.json"):
        if f.exists():
            s.update(json.loads(f.read_text()))
    return s


def data_root(override: str | None = None) -> Path:
    if override:
        p = Path(override)
        return p if p.is_absolute() else (CYCLE / p)
    for p in (VENDOR / "data" / "train", SYNTHETIC):
        if p.exists() and any(p.glob("*.small.features.parquet")):
            return p
    raise FileNotFoundError(
        f"No data for cycle {CYCLE_ID}: unpack the competition package into {VENDOR} "
        "(python common/tools/new_cycle.py ...) or run `python common/tools/make_synth.py` for a synthetic test panel."
    )


def is_synthetic(root: Path) -> bool:
    return "synthetic" in Path(root).parts


def ensure_predictor_importable() -> Path:
    """Put the vendor predictor.py of the active cycle on sys.path, else the local stub."""
    path = VENDOR if (VENDOR / "predictor.py").exists() else STUB
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def ensure_root_importable() -> None:
    if str(COMMON) not in sys.path:
        sys.path.insert(0, str(COMMON))


def archive_existing(path) -> Path:
    """Never lose an earlier output: if `path` exists, MOVE it to <dir>/_history/<stem>.<old-mtime-stamp><suffix>
    before the caller writes a new version. Returns the (now free) path."""
    from datetime import datetime

    p = Path(path)
    if p.exists() and p.is_file():
        stamp = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y%m%dT%H%M%S")
        hist = p.parent / "_history"
        hist.mkdir(parents=True, exist_ok=True)
        dst = hist / f"{p.stem}.{stamp}{p.suffix}"
        k = 1
        while dst.exists():
            dst = hist / f"{p.stem}.{stamp}_{k}{p.suffix}"
            k += 1
        p.replace(dst)
    return p
