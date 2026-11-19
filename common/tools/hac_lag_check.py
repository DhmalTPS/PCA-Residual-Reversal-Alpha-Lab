"""HAC-lag sensitivity of the headline metrics (no registry writes).

The IC and PnL series are hourly, but the target is a 120 h forward return, so neighbouring
observations overlap by up to 119 h. A Newey-West lag below the overlap understates the
long-run variance and overstates t-stats. This re-runs the official per-period evaluation of a
strategy with several lags and prints IC t, Sharpe +- s.e. and DSR side by side.

Usage (from the repo root, organiser package unpacked under vendor/):
    ALPHANOVA_CYCLE=S1/C4 python common/tools/hac_lag_check.py seasons/S1/C4/strategies/s09_pca_idio_reversal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import lab.paths as paths  # noqa: E402
from lab import registry  # noqa: E402

LAGS = [48, 120, 127]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    base = paths.settings()
    # read-only run: never touch the experiment registry or run store
    registry.append = lambda *a, **k: None
    registry.save_run = lambda *a, **k: "(not saved)"
    import evaluate as ev  # noqa: E402

    rows = []
    for lag in LAGS:
        s = dict(base, nw_lag_hours=lag)
        ev.settings = lambda s=s: s
        captured = {}
        orig = ev.summarize_folds

        def spy(*a, **k):
            m = orig(*a, **k)
            captured.update(m)
            return m

        ev.summarize_folds = spy
        ev.main([argv[0], "--mode", "full"])
        ev.summarize_folds = orig
        rows.append((lag, captured.get("IC"), captured.get("IC_t"), captured.get("SR_net_5bp"),
                     captured.get("SR_se_5bp"), captured.get("DSR"), captured.get("n_trials")))
    print("\n| NW lag (h) | IC | IC t | SR 5bp | s.e. | DSR | N |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join("%.4g" % v if isinstance(v, float) else str(v) for v in r) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
