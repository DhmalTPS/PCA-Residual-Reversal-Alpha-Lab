# S1/C4 - cycle summary

* Window: 2026-10-01 .. 2026-10-15 05:30 IST. Package: downloads/files.zip (31 periods) + city_coords.zip.
* Full decision log (newest first): [STATUS.md](STATUS.md). File moves during the 2026-10-08 restructure: [MIGRATION_MANIFEST.md](MIGRATION_MANIFEST.md).
  Earlier project notes: [PROJECT_NOTES.md](PROJECT_NOTES.md).

## Outcome
| item | result |
|---|---|
| Uploaded | `submissions/uploaded/s09_pca_idio_reversal.py` (sha 42e5f14f13d6ed58) -> signal_a0da8ba3ca2c4d29 |
| Server | novel (global = quality-gate 61.25 deg), status OK (passed the real overfitting gate), upload-time SR 0.0241 |
| Board | Sharpe 0.0028 (row 64), IC 0.0304, IC std 0.2187, conc 0.108 - not a prize place |
| Not novel (never onboarded) | `submissions/not_novel/`: S06 (16.3 deg, pot), S07 v1 (25.1), S07 v2 (38.4), S08 (42.3) |
| Ready, not uploaded | `submissions/ready_not_uploaded/`: S09-perp (rho 0.89 with S09 -> would self-block) |

## What we learned (measured, reusable)
1. Each period file is a hindsight reconstruction: the oldest rows show the opposite sign of directional relations
   (reports/r07_recency.md). Only the official per-period protocol is honest (lab/official.py, lab/honest.py).
2. Target = xs z-score of the 120h forward return sum starting 1h ahead (exact on scored rows). Hourly Sharpe ~ 0.35 x IC.
3. Gate emulation (joint row permutation + recomputed target, judged on Sharpe) reproduced the vendor demo pass and S09's pass.
4. Server data: S06's pot occupant (rho 0.96) passed the real gate -> our emulation is too strict for reversal signals.
5. S09 dev SR +0.026 but vault (030-031) IC -0.087 (t -2.5) and board 0.0028: the edge decayed/reversed in recent data.
   The vault was not run before upload - process error; now a hard rule.

## Where things are
* Strategies S01-S09: `strategies/` (S01-S03 synthetic-era, S04-S09 real-data).
* Research: `research/r00..r18*.py`, multi-agent outputs `research/agents/` (workflow wf_11551755-bb9).
* Every trial: `experiments/registry.csv`; runs with OOS predictions: `experiments/runs/<id>/`; vault log: `experiments/vault_log.csv`.
* Board snapshots + all server verdicts: `config/leaderboard.json`.
