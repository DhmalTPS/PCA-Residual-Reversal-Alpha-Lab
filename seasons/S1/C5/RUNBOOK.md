# S1/C5 - runbook for the first 6-7 hours (opens 2026-10-15 05:30 IST)

All commands from the project root in PowerShell with the venv active (`.\.venv\Scripts\Activate.ps1`).
Rules: PREREGISTRATION.md. Log every decision in STATUS.md (newest first).

| T+ | step | command / action | done when |
|---|---|---|---|
| 0:00 | Download | From the C5 competition page: the data package zip and the city-coordinates zip (verify email login) | both zips in Downloads |
| 0:10 | Install | `python common\tools\new_cycle.py S1/C5 --install --zip "<files.zip>" --zip "<city_coords.zip>" --activate` | `python common\tools\set_cycle.py` shows `* S1/C5` |
| 0:15 | Smoke | `python common\tools\run_vendor.py seasons\S1\C5\vendor\demo_engineered.py` | de-mean PASS |
| 0:20 | Copy templates | copy `common\research_templates\*` into `seasons\S1\C5\research\` (templates stay pristine) | files present |
| 0:25 | Audit | `python common\tools\run_research.py seasons\S1\C5\research\r00_data_audit.py` | periods, rows, NEW period ids noted in STATUS |
| 0:35 | Target | `... r01_target_identification.py -- --write` | horizon still 120h? (same season -> expected) |
| 0:45 | H1 / H2 | `lab.honest.evaluate` per PREREGISTRATION (one call each) + fresh-period SR | results in STATUS |
| 1:15 | Screens | `run_research.py ...\r06_official_screen.py`, `r09_stability_screen.py --symmetric`, `r16_x_alpha_screen.py` | shortlist |
| 2:30 | Candidates | build in `seasons\S1\C5\strategies\` from `common\strategy_templates\`; `evaluate.py --mode full` | admissible_v2 checked |
| 3:30 | Gates | `precheck.py` -> `dynamic_check.py` -> `shuffle_gate.py --reps 30` -> `vault.py` -> `city.py RUN_ID` | all logged |
| 4:30 | Promote | `freeze.py promote ...` -> `submissions\` | checklist written |
| 4:45 | Novelty | upload with "Check novelty first" (max 3) | verdicts in config\leaderboard.json |
| 5:30 | Decide | onboard only novel + passing; else submit nothing | STATUS + JOURNAL.md updated |
| 6:00 | Buffer | re-check board; record upload-time score; never project rank from it | |

Known pitfalls (from C4): `runner.py --full/--gauge-fix` print vault periods -> only after vault.py; Windows console
needs PYTHONIOENCODING=utf-8 (run_vendor.py handles it); the history hook must stay content-based (row hashes).
