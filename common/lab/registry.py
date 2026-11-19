"""Append-only experiment registry. Every evaluation is a trial (needed for the deflated Sharpe)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from lab.paths import REGISTRY, RUNS

COLUMNS = [
    "run_id", "utc", "strategy", "file_sha", "overrides", "mode", "data", "hours",
    "IC", "IC_se", "IC_t", "SR_gross", "SR_net_5bp", "SR_se_5bp", "SR_ci90_lo_5bp", "SR_ci90_hi_5bp",
    "SR_net_10bp", "SR_shrunk", "P_SR_pos", "PSR0", "DSR", "SR_star", "n_trials",
    "months_pos_frac", "months_t", "turnover_mean", "cost_drag_sd_units", "IC_disp_corr",
    "concentration_raw", "flat_fraction", "longest_flat_run", "train_s_max", "predict_s_max", "notes",
]


def load() -> pd.DataFrame:
    if REGISTRY.exists():
        return pd.read_csv(REGISTRY)
    return pd.DataFrame(columns=COLUMNS)


def n_trials(data_tag: str) -> int:
    """Trials so far on this data (+1 for the current one)."""
    reg = load()
    return int((reg["data"] == data_tag).sum()) + 1 if len(reg) else 1


def sr_sd_trials(data_tag: str) -> float | None:
    """Cross-trial spread of SR estimates, from COMPARABLE trials only (mode 'full' evaluations).
    Research-screen rows (raw-sign constructions with very different turnover) are counted in N but
    excluded here: their spread is not the null spread of a submission-like signal. With < 10
    comparable trials return None -> the caller uses the candidate's own s.e. (the null spread
    when every trial is noise)."""
    reg = load()
    if not len(reg):
        return None
    m = (reg["data"] == data_tag) & (reg["mode"] == "full")
    v = pd.to_numeric(reg.loc[m, "SR_net_5bp"], errors="coerce").dropna()
    return float(v.std(ddof=1)) if len(v) >= 10 else None


def new_run_id(strategy: str) -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{strategy}"


def append(row: dict) -> None:
    """Append ONE line (no read-modify-write), so concurrent runs cannot clobber each other."""
    import csv

    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    new = not REGISTRY.exists()
    vals = []
    for k in COLUMNS:
        v = row.get(k, "")
        if isinstance(v, float) and not np.isfinite(v):
            v = ""
        vals.append(v)
    with open(REGISTRY, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        w.writerow(vals)


def save_run(run_id: str, P: pd.DataFrame, metrics: dict, meta: dict) -> str:
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=True)
    P.to_parquet(d / "preds.parquet")
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json))
    (d / "meta.json").write_text(json.dumps(meta, indent=2, default=_json))
    return str(d)


def latest_runs(mode: str = "full", data_tag: str | None = None) -> pd.DataFrame:
    """Most recent run per (strategy, overrides)."""
    reg = load()
    if not len(reg):
        return reg
    reg = reg[reg["mode"] == mode]
    if data_tag:
        reg = reg[reg["data"] == data_tag]
    reg = reg.fillna({"overrides": ""})
    return reg.sort_values("utc").groupby(["strategy", "overrides"], as_index=False).tail(1)


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
