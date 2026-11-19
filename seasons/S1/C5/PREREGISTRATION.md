# S1/C5 - pre-registration (written 2026-10-09, BEFORE any C5 data)

Purpose: fix hypotheses and decision rules in advance so the C5 window cannot turn into data-snooping.
Everything below was decided from S1/C4 evidence only. Do not edit after the C5 package arrives; add a dated
"Amendments" section instead (with the reason).

## What C4 taught us (the reason for these rules)
* Board Sharpe is scored on a hidden band MORE RECENT than the shipped data. S09 scored dev +0.026 but board 0.0028:
  its 4-week smoothing lagged the recent regime (vault positions IC t -2.5) although the raw signal stayed positive.
* Recent-period behaviour matters more than full-history significance.

## The bar (lab.honest.admissible_v2) - applied to C5's own data
1. Full dev history: SR5 >= 0.0133 (or the C5 board's 3rd place, whichever is higher), ci_lo > 0, SR10 > 0.008,
   periods_pos >= 0.65, DSR >= 0.5 (honest trial count), max |cos| with every occupied leg <= 0.35.
2. Recent: SR5 over the last 8 dev periods > 0.
3. Vault: net SR5 over C5's vault periods >= 0.
4. FRESH-period check: net SR on the period(s) that are NEW in the C5 package (not in C4's 31 files) must be > 0.
5. Gate emulation (shuffle_gate --reps 30) PASS, or a documented reason it is too strict for that signal type.
6. Server "Check novelty first" = novel. Never onboard a not-novel signal.
A candidate that fails ANY of 1-5 does not get a novelty check.

## Occupied space entering C5 (blocks similar ideas)
Legacy pot will contain every C4 accepted signal, including OUR S09 (seasons/S1/occupied_directions.py lists
S06, S07, S08, S09). A C5 candidate must be <= 0.5 correlated with S09 as submitted (hl 672).

## Pre-registered hypotheses (test each ONCE on C5 data; log every evaluate() call)
| id | hypothesis | construction | C4 evidence | C5 pass rule |
|---|---|---|---|---|
| H1 | Recent-regime TIMING of factor-residual reversal | `residualize(unit(smooth(S09_raw, 48)), [unit(smooth(S09_raw, 672))])`, hl=0 | dev SR +0.0015 (no full-history edge), recent +0.021, vault +0.060, cos with S09 0.000 | full bar 1-6. Expected to FAIL bar 1 (dev); listed so that a pass on C5 is not post-hoc. |
| H2 | Faster S09 is a different enough signal | `S09_raw` smoothed hl 48 | dev +0.019, recent +0.034, vault +0.042, but cos with submitted S09 0.547 (> 0.5) | only if C5-measured cos with S09 <= 0.45 AND bar 1-6 |
| H3 | Fresh screen | research templates r06 / r09 / r16 on C5 (copied into C5/research) | n/a | bar 1-6 for any candidate; at most 3 novelty checks |

## Stop rules
* If no candidate passes bar 1-5 by T+5h, submit NOTHING (every passing submission enters the pot; a weak one only
  blocks our own future ideas).
* Max 3 novelty checks in C5 unless a check reveals new geometry worth one more (document why).


## Amendments
### A1 - 2026-10-15 11:05 IST (after installing the C5 package; no strategy evaluated yet)
* FACT: the C5 data package is byte-identical to C4 (files.zip sha256 34f22555ed680065..., 174 identical entries,
  31 periods). Only the city map changed. => Rule 4 ("fresh period > 0") is IMPOSSIBLE: there is no unseen local data.
* FACT: C5 is the LAST cycle of Season 1; the pot is cleared at the season boundary. Accepted C5 submissions therefore
  cannot block our future work; within C5 the greedy rule simply stars the best of our mutually-correlated entries.
* FACT (rules): the live window is ~2% of scored hours, so the board Sharpe of a submission is ~98% of its final score.
  Board feedback is a direct measurement of the objective, not an overfitting channel (only the live 2% is unseen).
* CHANGES: (a) rule 4 is replaced by a RECENCY PROXY: net SR over the 10 most recent periods (dev 022-029 + vault
  030-031), because S09's board result (0.0028) matched its vault (-0.010) and not its full-history SR (+0.026);
  (b) the submission budget is used as measurement: up to ~12 honest, gate-passing, mutually |rho| <= 0.5 candidates
  may be uploaded over the cycle, ranked by the recency proxy, then judged by their board Sharpe; (c) bars 1-3, 5, 6 stay.

## Deviations (recorded 2026-11-19, after the cycle closed; not a contemporaneous amendment)
* Rule "A candidate that fails ANY of 1-5 does not get a novelty check" and A1(c) were not followed: A (full SR
  +0.0111), AB (+0.0125) and L240 (+0.0064) failed bar 1 (SR5 >= 0.0133) and were novelty-checked anyway, treated as
  measurements under A1(b). Only Z (the 4th check) met the bar.
* Stop rule "Max 3 novelty checks in C5": 4 checks were used. The reason for the 4th (the blocker model was refuted
  and Z was the first candidate to pass admissible_v2) is in STATUS.md, but no amendment was written before it.
* All four checks came back not novel (59.18, 50.38, 50.78, 58.14 deg); nothing was submitted.
