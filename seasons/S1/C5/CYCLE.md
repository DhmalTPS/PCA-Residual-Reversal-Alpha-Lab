# S1/C5 - cycle notes

* Opened: 2026-10-09
* Package: see downloads/ (originals) and vendor/ (as shipped - never edit vendor files)
* Status / decision log: STATUS.md (append newest first)
* Outcome: no submission; 4 novelty checks, all not novel (59.18, 50.38, 50.78, 58.14 deg); factor-residual reversal
  family saturated. Three of the four checks deviated from the pre-registration (see PREREGISTRATION.md, "Deviations").

## First steps (proven workflow, see <ROOT>/README.md)
1. python common/tools/set_cycle.py S1/C5
2. python common/research_templates/r00_data_audit.py ; r01_target_identification.py --write
3. honest per-period screens (r06 / r09 / r16 templates), lab.honest.evaluate for every candidate
4. gates: precheck -> dynamic_check -> shuffle_gate -> VAULT (always, before any server check) -> freeze promote
