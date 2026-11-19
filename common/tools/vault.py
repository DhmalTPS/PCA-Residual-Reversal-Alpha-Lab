"""One-shot vault score: train on all dev data (minus embargo), predict the held-out vault months.

  python tools/vault.py strategies/s02_lgbm_regime.py

The vault is the last `vault_months` months, never used for any decision before this call.
Each file version (sha) may be vaulted ONCE; a second attempt is refused unless --force, and
forced runs are logged as such (looking twice turns the vault into another dev set).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from lab.io import load_period  # noqa: E402
from lab.loader import file_sha, load_strategy  # noqa: E402
from lab.metrics import summarize_folds  # noqa: E402
from lab.official import periods_split, run_official, val_only  # noqa: E402
from lab.paths import REPORTS, VAULT_LOG, archive_existing, data_root, is_synthetic, settings  # noqa: E402
from lab.walkforward import run as walk  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--data", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    s = settings()
    root = data_root(a.data)
    tag = "synthetic" if is_synthetic(root) else "real"
    sha = file_sha(a.file)
    log = pd.read_csv(VAULT_LOG) if VAULT_LOG.exists() else pd.DataFrame(columns=["utc", "file", "sha", "data", "forced", "SR_net_5bp", "SR_se_5bp", "IC"])
    seen = log[(log["sha"] == sha) & (log["data"] == tag)]
    if len(seen) and not a.force:
        print(f"REFUSED: {Path(a.file).name} (sha {sha}) was already vaulted on {seen.iloc[-1]['utc']}. Use --force only if you accept a contaminated vault.")
        return 2
    cls = load_strategy(a.file)
    _, vault_p = periods_split(s, root)
    print(f"vault periods {vault_p} (official per-period protocol)")
    m = summarize_folds(val_only(run_official(cls, s, vault_p, root=root)), s)
    row = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "file": Path(a.file).name, "sha": sha,
           "data": tag, "forced": bool(a.force and len(seen)), "SR_net_5bp": m["SR_net_5bp"], "SR_se_5bp": m["SR_se_5bp"], "IC": m["IC"]}
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    VAULT_LOG.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(VAULT_LOG, index=False)
    REPORTS.mkdir(exist_ok=True)
    archive_existing(REPORTS / f"vault_{Path(a.file).stem}_{sha}.json").write_text(json.dumps(m, indent=2, default=str))
    print(f"VAULT {Path(a.file).name} ({m['hours']} h): SR@5bp {m['SR_net_5bp']:+.4f} +- {m['SR_se_5bp']:.4f} "
          f"[90% CI {m['SR_ci90_lo_5bp']:+.4f}, {m['SR_ci90_hi_5bp']:+.4f}], SR@10bp {m['SR_net_10bp']:+.4f}, IC {m['IC']:+.4f} (t {m['IC_t']:.2f})")
    print("Decision rule: the vault must agree in SIGN with the dev estimate and lie inside its CI; "
          "a 2-period vault (1600 h) has SE ~ 0.025, so it can veto, not confirm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
