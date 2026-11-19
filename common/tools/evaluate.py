"""Walk-forward evaluation of one strategy file -> metrics with uncertainty + registry row.

  python tools/evaluate.py strategies/s01_ridge_xs.py --mode full
  python tools/evaluate.py strategies/s01_ridge_xs.py --mode quick --set SMOOTH_HALFLIFE=12 --set BAND=0.3

Every call is a TRIAL and is logged (needed for the deflated Sharpe). Vault months are excluded
(use tools/vault.py for the one-shot vault score).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from lab import registry  # noqa: E402
from lab.io import load_period  # noqa: E402
from lab.loader import file_sha, load_strategy, parse_overrides, with_overrides  # noqa: E402
from lab.metrics import summarize, summarize_folds  # noqa: E402
from lab.official import periods_split, run_official, stacked, val_only  # noqa: E402
from lab.paths import data_root, is_synthetic, settings  # noqa: E402
from lab.walkforward import run as walk  # noqa: E402

HEADLINE = [
    ("SR_net_5bp", "net Sharpe @5bp"), ("SR_se_5bp", "  NW s.e."), ("SR_ci90_lo_5bp", "  90% CI lo"),
    ("SR_ci90_hi_5bp", "  90% CI hi"), ("SR_net_10bp", "net Sharpe @10bp (stress)"), ("SR_gross", "gross Sharpe"),
    ("SR_shrunk", "shrunk Sharpe (prior tau)"), ("P_SR_pos", "P(true SR > 0)"), ("PSR0", "PSR(0) [T_eff]"),
    ("DSR", "deflated SR prob"), ("SR_star", "  SR* (max of N nulls)"), ("n_trials", "  trials N"),
    ("IC", "IC (vs target)"), ("IC_t", "  IC t-stat (NW)"), ("IC_std", "  IC std (board col 3)"), ("IC_disp_corr", "IC-dispersion corr"),
    ("months_pos_frac", "months positive"), ("months_t", "t across months"),
    ("turnover_mean", "turnover ||dP||_1"), ("cost_drag_sd_units", "cost / gross sd"),
    ("concentration_raw", "concentration (ticker-frame proxy)"), ("flat_fraction", "flat fraction"),
]


def bar_report(m: dict) -> list[str]:
    from lab.paths import CONFIG
    lb_path = CONFIG / "leaderboard.json"
    if not lb_path.exists():
        return []
    lb = json.loads(lb_path.read_text())
    from statistics import NormalDist
    mu, sd = m.get("SR_shrunk"), m.get("SR_post_sd")
    if not (mu is not None and sd and np.isfinite(mu) and np.isfinite(sd)):
        return []
    N = NormalDist(mu, sd)
    out = [f"vs public board ({lb['as_of_utc'][:10]}), using the shrunk posterior N({mu:+.4f}, {sd:.4f}):"]
    for k in ("sharpe_third", "sharpe_second", "sharpe_first"):
        out.append(f"   P(true SR > {k.split('_')[1]:<6} {lb[k]:.4f}) = {1 - N.cdf(lb[k]):.3f}")
    out.append("   (local Sharpe != board Sharpe: hidden band + extra official cost adjustments; calibrate after the first submission)")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--mode", choices=["quick", "full"], default="full")
    ap.add_argument("--set", action="append", default=[], help="class attribute override KEY=VALUE")
    ap.add_argument("--retrain-every", type=int, default=1, help="retrain every k test months")
    ap.add_argument("--data", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--protocol", choices=["official", "singlefile"], default="official",
                    help="official = per-period files (honest); singlefile = latest file only (BIASED: hindsight in the reconstruction)")
    a = ap.parse_args(argv)

    s = settings()
    root = data_root(a.data)
    data_tag = "synthetic" if is_synthetic(root) else "real"
    overrides = parse_overrides(a.set)
    cls = with_overrides(load_strategy(a.file), overrides)
    stem = Path(a.file).stem
    print(f"evaluate {stem} mode={a.mode} data={data_tag} overrides={overrides or '-'}")
    if data_tag == "synthetic":
        print("  NOTE: synthetic panel -> tool validation only; these numbers say nothing about the real target.")

    if a.protocol == "official":
        data_tag = data_tag + "-official"
        dev_p, _ = periods_split(s, root)
        if a.mode == "quick":
            dev_p = dev_p[len(dev_p) - int(s["quick_test_months"]):]
        folds = val_only(run_official(cls, s, dev_p, root=root))
        P = stacked(folds, "P")
        n = registry.n_trials(data_tag)
        m = summarize_folds(folds, s, n_trials=n, sr_sd_trials=registry.sr_sd_trials(data_tag))
        info = [{k: f[k] for k in ("period", "train_rows", "train_s", "predict_s")} for f in folds]
        panel = type("P", (), {"period": dev_p[-1]})()
    else:
        print("  WARNING: singlefile protocol is optimistic on the shipped data (per-period reconstruction).")
        panel = load_period(root)
        P, info = walk(cls, panel, s, mode=a.mode, retrain_every=a.retrain_every)
        n = registry.n_trials(data_tag)
        m = summarize(P, panel, s, n_trials=n, sr_sd_trials=registry.sr_sd_trials(data_tag))
    m["folds"] = info
    m["train_s_max"] = max(f["train_s"] for f in info)
    m["predict_s_max"] = max(f["predict_s"] for f in info)

    run_id = registry.new_run_id(stem)
    meta = {"file": str(Path(a.file).resolve()), "file_sha": file_sha(a.file), "overrides": overrides,
            "mode": a.mode, "data": data_tag, "period": panel.period, "settings": s}
    path = registry.save_run(run_id, P, m, meta)
    row = {k: m.get(k) for k in registry.COLUMNS}
    row.update({"run_id": run_id, "utc": run_id.split("_")[0], "strategy": stem, "file_sha": meta["file_sha"],
                "overrides": json.dumps(overrides, sort_keys=True) if overrides else "", "mode": a.mode,
                "data": data_tag, "notes": a.notes})
    registry.append(row)

    print(f"\n  run {run_id}  ({m['hours']} OOS hours, {m['months']} months)  -> {path}")
    for k, label in HEADLINE:
        v = m.get(k)
        if isinstance(v, float):
            print(f"  {label:<36} {v:+.4f}")
        elif v is not None:
            print(f"  {label:<36} {v}")
    print("  IC decay (lag h -> IC):", {k: round(v, 4) for k, v in m["IC_decay"].items()})
    for line in bar_report(m):
        print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
