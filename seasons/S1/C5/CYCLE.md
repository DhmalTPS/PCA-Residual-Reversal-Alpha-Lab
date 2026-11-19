# S1/C5 - cycle notes

* Opened: 2026-10-08
* Package: see downloads/ (originals) and vendor/ (as shipped - never edit vendor files)
* Status / decision log: STATUS.md (append newest first)
* Outcome: (fill at close: submissions, server verdicts, board rank, lessons) and add one line to <ROOT>/JOURNAL.md

## First steps (proven workflow, see <ROOT>/README.md)
1. python common/tools/set_cycle.py S1/C5
2. python common/research_templates/r00_data_audit.py ; r01_target_identification.py --write
3. honest per-period screens (r06 / r09 / r16 templates), lab.honest.evaluate for every candidate
4. gates: precheck -> dynamic_check -> shuffle_gate -> VAULT (always, before any server check) -> freeze promote
