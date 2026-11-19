# STATUS / HANDOFF - updated 2026-10-08 (S09 accepted but board SR 0.0028: recent regime reversal - see Session 3 top entry)

Deadline: S1C4 submissions close **15 Oct 2026 05:30 IST (00:00 UTC)**. Target: submit 1-3 distinct
candidates by ~13 Oct, >= 24 h before close. Read PROJECT.md for layout/workflow; this file is the state.

## Environment
* venv: `.venv` (Python 3.13). Runner on Windows needs `PYTHONIOENCODING=utf-8` and `COMP6_DISABLE_PIP_INSTALL=1`.
* Vendor package unpacked into the root as shipped (never edit vendor files). `downloads/` holds the original zips.
* `signal_db/signal_cities_ambient_coords.parquet` = merge of downloaded map (1628) + shipped-only (18) = 1646 cities;
  originals kept beside it (`.shipped.parquet`, `.downloaded_2026-10-06.parquet`).
* No background jobs running. `_incoming/` only holds the extracted city zip (deletable).

## Session 2 (2026-10-07, resumed) - new findings, newest first
* **C4 CLOSED BY DECISION (2026-10-07):** novelty check #4 S08 NOT NOVEL (global = quality-gate 42.34 deg, rho 0.74;
  predicted 41.6..74.2). Discarded. Pre-registered stop rule applied: no further C4 checks (4/10 used, 0/20 submissions,
  nothing of ours in the legacy pot). All four honest edges are occupied: level reversal (pot, 16 deg), extremeness
  (current, 25/38 deg), F5+5-day reversal (current, 42 deg). Next opportunity: C5 (same target; C4 occupants move into
  the pot -> same directions stay blocked) or Season 2 (new target, pot cleared) - re-run r01/r06/r07/r09/r16 first.
* S08 vault (one shot, partially contaminated - its legs are inside S06 whose vault was seen): SR +0.0080 +- 0.0223,
  IC -0.007 -> not vetoed, not confirmed. Gate calibration point #2: S06's pot occupant (rho 0.96 with S06) PASSED the
  real gate, while our full-slice emulation fails S06 at p=1.0 -> emulation likely too strict for reversal signals.
  Recommendation: novelty check #4 on submit/s08_two_leg_reversal.py; novel -> onboard (gate failure costs no slot);
  not novel -> discard and stop C4.
* **S08 two-leg reversal READY (submit/s08_two_leg_reversal.py):** unit(-rank F5) + unit(-mean rank d120 F1-4),
  hl 672, FIXED. Honest dev SR +0.0216 [+0.0115,+0.0320], @10bp +0.0144, IC 0.039 t4.0, 83% periods +, DSR 0.73 (N=279),
  P(>3rd 0.0133) 0.75 (shrunk). rho_hat with S06 0.53 (57.9 deg -> 41.6..74.2 deg from S06's pot occupant: novelty
  ~coin flip), with S07 ~0. Gate emulation FAIL (in-sample -0.0155 vs null -0.0070) - calibrated on one point only;
  a server gate failure costs no slot. Perp-S06 versions destroy the edge (SR -0.013). r16: 180-construction X-alpha
  screen -> 0 candidates pass SR + gate + low-overlap; the gate is the binding constraint for non-occupied edges.
  S07 v2 moved to the not-novel list (still in submit/ - do not re-upload).
* **Regime work (r14/r15):** S06 is in-sample negative/zero in EVERY regime half (cannot be gate-safe anywhere);
  S07's edge concentrates in HIGH xs dispersion of 24h changes (gate t 1.45 vs 0.31; val t 4.2 vs 0.85).
  Time-share S07(high)/S06(low), 8 pre-declared variants: raw regime -> ~90 switches/1000h, SR -0.04 (cost);
  smoothed regime -> SR +0.001..+0.006, gate in-sample ~ null, est rho_max 0.48-0.53. REJECTED.
* r16 running: screen oriented by honest net Sharpe on X (the scored quantity), raw + perp(S06,S07) versions,
  with gate-slice in-sample SR and overlap with both legs. ~180 trials logged.
* **Novelty check #3: S07 v2 NOT NOVEL** - global = quality-gate 38.40 deg (rho 0.78; v1 was 25.1 deg). Discard.
  Local geometry (reports/r13_probe_design.csv): S07 ingredients are 0.83-0.97 correlated; moving further from the
  occupant (DD-heavy / DD perp v1 / 2DD-v1, up to 58 deg from v1) keeps SR ~0.008-0.011 but the IC on the target
  turns ~0 or negative (t -1.5) -> edge evidence becomes inconsistent; P(top 3, bar now 0.0133) ~ negligible.
  Conclusion for C4: both robust edges (reversal, extremeness) are occupied; no novel candidate with a credible
  prize chance exists in the same-hour / two-point universe. 7 checks and 20/20 submissions remain.
* **S07 v2 READY (sha a34eb36d828ee554), awaiting server novelty check #3:** members |z| levels + |z| d24 + |z| d120
  of F1-4 (12, equal weight), hl 672 / window 2688 / HISTORY 3000, O(T) exact recursive smoother (predict 2.1 s on
  49k rows; was 41 s). Honest dev SR +0.0109 +- 0.0067 [-0.0006,+0.0223], @10bp +0.0092, IC 0.027 t2.7, 69% periods +,
  P(SR>0) 0.92, P(>3rd) 0.32 (shrunk), DSR 0.20 (N=84). Gate emulation PASS z 7.1. Dev city 48 deg.
  Rationale: S07 v1 was blocked by a CURRENT-cycle signal (global = quality-gate = 25.12 deg; stats match board Q6,
  SR 0.0075); greedy admission by Sharpe => a higher-Sharpe S07 should take that slot. S06's blocker is in the
  legacy pot (quality-gate 16.31 < global 35.25) -> cannot be displaced.
  r11 (orthogonal screen): after projecting out S06+S07 per hour no robust edge remains (best |t| 1.4 of 72).
  r12: 8 pre-declared S07 variants logged; hl 672 adds ~+0.003 in every variant. Not vaulted (vault spent).
  submit/discarded/ holds the not-novel v1 + S06 files (do not upload).
* **SERVER NOVELTY CHECKS (2 of 10 used, 0/20 submissions):** S07 "not novel" global 25.12 deg / quality-gate
  25.12 deg (rho 0.91); S06 "not novel" global 35.25 / quality-gate 16.31 deg (rho 0.96). Both DISCARDED (not onboarded).
  => both honest edges are occupied. New candidates must be >= 76 deg from S06 and >= 85 deg from S07 (local
  |rho| <= 0.24 / 0.09) to be guaranteed >= 60 deg from the occupants. r11_orthogonal_screen.py projects S06/S07
  out per hour and screens what edge remains. Upload page has "Check novelty first" (use it for every candidate).
* **READY TO SUBMIT (2026-10-07):** `submit/s07_extremeness.py` and `submit/s06_reversal_composite.py`
  (both FIXED weights, hl 336, window 1344, HISTORY 1600; static + runtime gates re-run on the exact files;
  vendor quick check de-mean PASS). Vault (ONE shot, logged in experiments/vault_log.csv): S06 +0.0067 +- 0.0185,
  S07 +0.0083 +- 0.0184 (IC +0.034) -> neither vetoed (pre-registered rule: veto if < -1 s.e.).
  Vendor --gauge-fix AFTER the vault: S06 SR +0.0184, IC 0.022, conc 0.385, city 24.1 deg; S07 SR +0.0064,
  IC 0.017, conc 0.159, city 43.9 deg. Gate emulation: S07 PASS (z 5.8), S06 FAIL (p 1.0).
* Recommendation: upload S07 (gate-safe, sparse region, likely quality signal, probably below
  3rd on Sharpe) and S06 (honest probe: strong if the real gate admits it; a gate failure costs nothing - no cap,
  no pot). After upload, record the server verdicts + board global novelty in config/leaderboard.json: they
  calibrate the gate emulator (S06 verdict is the key datum).
* Disclosure: vendor --full on demo/S04 (before the vault concept) printed their 030/031 Sharpes; S06/S07 were not
  designed on them.
* **Gate slice resolved (best-calibrated):** vendor demo PASSES only with the FULL training slice (z 7.3);
  fails at 730 rows (p .16) and 5000 rows (p .065). => full-slice emulation is the calibrated one, and under
  it S06 FAILS (p 1.0). Failed submissions do not count against the cap and do not enter the pot ->
  S06 can be submitted as a free, honest probe of the real gate.
* **S07 extremeness** (strategies/s07_extremeness.py, FIXED, 8 inputs |z| levels + |z| 24h changes of F1-4,
  hl 336): honest dev SR@5bp +0.0070 +- 0.0064, P(SR>0) 0.83, 66% periods +, DSR 0.09 (N=75). City 46.7 deg,
  only 13/1646 cities within 60 deg (sparse). rho_hat(S06,S07) = -0.022 (orthogonal). Learned mode worse.
  Gate emulation running (job bw8luam3b).
* DSR fix: registry.sr_sd_trials now uses only mode='full' runs (else candidate s.e.). Honest DSR (N=75):
  S06 0.76, S07 0.09.
* Plan: (1) S07 gate result; (2) vault ONCE for final S06 (hl336) and S07; (3) promote both via
  tools/freeze.py; (4) recommend submitting S07 (gate-safe, novel) + S06 (probe; strong if the real gate
  passes it). Expected ranks: S06 ~top-3 if admitted and not blocked; S07 likely below 3rd (0.011).
* **Gate reference-slice uncertainty (being measured):** emulator with the FULL training slice of file 029
  rejects S06 (in-sample SR -0.0152 vs null -0.0059 +- 0.0015, p=1.0) because old in-file rows have the
  reversed sign. The rules' "training slice of one particular month" may mean ~730 rows; re-running the
  gate with --rows 730 / 5000 for demo_engineered and S06 (job bag9qtff1; results -> reports/shuffle_*.json,
  note the json is overwritten per run - read the console log).
* **Extremeness family** (r09/r10): -rank(zFa*zFb), a,b in F1..F4 (~ -rank|z|): the ONLY family with the same
  sign in the gate slice (IC +0.011, t 2.5) and validation (IC +0.020, t 3.7, 72% periods). Gate-friendly
  (in-sample SR +0.002) and an even function (should be ~orthogonal to level reversal), but net SR only
  +0.005 @5bp hl336 -> below the board bar alone.
* **S06 reversal composite** (strategies/s06_reversal_composite.py, FIXED weights, hl 336, window 1344,
  HISTORY 1600): honest dev SR@5bp +0.0192 [+0.009,+0.030] (hl168 run 20261007T141533), +0.0156 @10bp hl336,
  79% periods +, t across periods 3.8. Weights were chosen on the dev validation blocks (68 trials logged ->
  deflated hurdle ~0.012-0.015). City novelty 25.2 deg, 332 cities within 60 deg (CROWDED region).
  Learned version (FIXED=False, ridge on recent 6000 rows): -0.003 -> weights are not learnable in-file.
* **Within-file recency drift** (reports/r07_recency.md): oldest rows of every file have the OPPOSITE sign to
  the scored rows; recent rows the same sign. Explains S04's failure. MAX_TRAIN_ROWS added to S04/S05/S06.
* Honest screen r06 (FIR-smoothed, 48 s): levels F1-F4 IC -0.036 (t -4.2), F6 +0.017 (sign flips vs single
  file), d120 F1-4 -0.029, products -0.020. S05 recency-6000: SR -0.0065, IC +0.012 (too many inputs).
* tools/city.py: gauge city on DEV periods only (vault-safe); validated vs runner on S04 (32.3 vs 29.0 deg,
  same nearest). DO NOT run `runner.py --full/--gauge-fix` before the vault decision (prints 030/031).
* registry: 64 research-screen trials logged (mode=screen) -> honest DSR trial count (68 on real-official).

## Session 3 (2026-10-07/08) - dedicated multi-agent C4 push (user request)
* **2026-10-09 C5 preparation (post-cycle analysis on C4 data, logged as trials):** S09 by smoothing half-life -
  hl672 (submitted): dev +0.0259, recent(022-029) +0.0155, vault -0.0103; hl168: +0.0257 / +0.0262 / +0.0058;
  hl48: +0.0192 / +0.0341 / +0.0421 (SR10 only +0.006). Raw-signal vault IC +0.032 (t 1.7) vs positions IC -0.087:
  the 4-week smoothing lagged the recent regime. Fast S09 (hl48) is 0.547 correlated with submitted S09 (pot) -> blocked.
  Timing component (fast perp slow): dev ~0, recent +0.02, vault +0.04..0.06 -> recent-only effect, fails the bar.
  Harness: evaluate() now reports SR5_recent, vault_SR5, vault_IC_raw/_pos; admissible_v2 = v1 + recent > 0 + vault >= 0.
  new_cycle.py --install (install into the pre-created empty C5). Consistency audit: all checks OK.
  C5 rules written BEFORE C5 data: seasons/S1/C5/PREREGISTRATION.md, RUNBOOK.md.
* **2026-10-08 fixes after restructure:** (1) S01 row_shift fixed - exact trailing windows (sliding windows) replace pandas
  streaming rolling; values identical to pandas within 3e-14, row-shift diff 8.9e-5 -> 0.0; original kept at
  strategies/_history/s01_ridge_xs.original_pandas_rolling.py (registry rows for the old sha remain valid).
  (2) Season-specific occupied directions moved out of common/: seasons/S1/occupied_directions.py (S06, S07, S08 + S09,
  our accepted entry); lab/honest.py is season-driven (old version in common/lab/_history/; LEG_HL alias kept for archive scripts).
* **2026-10-08 RESTRUCTURE:** project reorganised into <ROOT>/common (reusable) + seasons/S1/C4 (this cycle); moves only
  (MIGRATION_MANIFEST.md). New tools: set_cycle, new_cycle, run_vendor, run_research (runs any old script in its cycle
  context, preserving overwritten outputs in _history/). Tools now archive previous outputs. Selftest 21/21 PASS.
  Finding during the migration test: legacy S01 (synthetic-era, never submitted) FAILS row_shift on real data (8.9e-5 >
  1e-6: pandas rolling numerics depend on the frame start) - all submitted/template strategies pass (0 / 1e-11).
* **S09 ON THE BOARD: Sharpe 0.0028 (row 64, not starred), IC 0.0304, global 61.25 deg.** Diagnosis: NOT cold-start/chunking
  (warm SR identical for 800/168/24-row predict chunks; cold 0.008-0.011). It is RECENT REGIME REVERSAL: dev per-period mean
  SR 001-010 +0.029, 011-020 +0.034, 021-029 +0.016; VAULT 030-031 (run post-upload) SR -0.0103 +- 0.0205, IC -0.087 (t -2.5).
  The board's scored history extends BEYOND our data (hidden recent band), where S09 loses. Board PnL chart: rises then gives
  back most. PROCESS ERROR: S09 was not vaulted before upload (vault wrongly treated as spent; it was clean for S09).
  Rule from now on: vault EVERY new candidate before any server check, and weight recent periods (last ~6 + vault) heavily.
  C4 recommendation: stop (no candidate credible vs 0.0133 on recent data). 5/10 checks, 19/20 submissions left.
* **S09 ACCEPTED (2026-10-08 10:44 IST): signal_a0da8ba3ca2c4d29, status OK, NOVEL global = quality-gate 61.25 deg
  (rho 0.481), server Sharpe 0.0241 (> board #1 0.0199 as of 10-07) -> projected Q1.** Calibration: server SR 0.0241 vs
  local warm 0.0259 / cold 0.0153 => server calls generate_features; gate emulation correct (pass). DO NOT upload
  S09-perp (rho 0.893 with S09 -> self-blocked, pot pollution). Risks: thin novelty margin (1.25 deg), new higher-SR
  correlated entrants before close. Next: confirm board rank; optional search for a 2nd signal orthogonal to S09 that
  could take 2nd/3rd (must be |rho| <= 0.5 vs S09 AND the occupants).
* Shared harness lab/honest.py (validated: reproduces S08 exactly). PRE-REGISTERED bar for spending a server novelty
  check (lab.honest.admissible): SR5 >= 0.0133, ci_lo > 0, SR10 > 0.008, periods_pos >= 0.65, DSR >= 0.5 (honest N),
  max |cos| with S06/S07/S08 legs <= 0.35. Candidates that clear it must ALSO survive 2/2 adversarial skeptics
  (causality, re-run, half-sample 001-014 vs 015-029 both positive, smoothing hl/2 and 2hl).
* Workflow c4-honest-edge-hunt (run wf_11551755-bb9): 5 families - timescale, neutralisation, offline_learned (leave-one-
  period-out on honest rows), peer_factor, f5f6_nonlinear; <= 12 evaluate() calls each (all logged as trials);
  agent code in research/agents/. Vault (030/031) and runner --full are off-limits.

* **Workflow results (wf_11551755-bb9):** timescale NULL (1-day reversal gross ~= 5bp cost); neutralisation NULL (on
  scored rows the target is EXACTLY the xs-z of the 120h fwd sum, cos 1.000; the 0.936 came from reconstructed old rows);
  offline_learned weak (nested-CV unbiased SR 0.0075); f5f6_nonlinear NULL (resumed after a network failure, 10/12 evals,
  best +0.0141 but IC t 1.0, H1 +0.003 / H2 +0.022 -> noise; the earlier F6 edge was its S06 overlap).
  **peer_factor: S09 = PCA-idiosyncratic 120h reversal of Feature.1 (K=8 factors, 720h trailing cov), hl 672.**
  Honest SR +0.0259 [+0.0166,+0.0352], @10bp +0.0190, IC t 4.8, 83% periods +, t across periods 4.5, DSR 0.93 (N=325);
  half-samples +0.025/+0.027; hl 336-1344 all +0.024..+0.027; causality audited by 2 skeptics (both refuted ONLY on
  novelty uncertainty). GATE EMULATION PASS (real -0.0116 vs null -0.0246 +- 0.0037, z 3.5). Dev city 53.5 deg, only
  6/1646 cities within 60 deg. rho_hat vs probes: S06 0.244 (>=59.6 deg from its pot occupant), S07v2 0.030,
  S08 0.264 (32..117 deg from S08's occupant: the open risk; typical-case ~78 deg).
  S09-perp (PERP_S08=True): SR +0.0147 [+0.0057], DSR 0.33, all |rho| <= 0.09 -> >= 49-72 deg from every occupant.
  READY: submit/s09_pca_idio_reversal.py (sha 42e5f14f13d6ed58) and submit/s09_pca_idio_reversal_perp_s08True.py
  (sha 946ae168b99a43b6). Pre-registered-rule amendment (stated openly): edge confirmed genuine by both skeptics; only
  novelty uncertain, which is what a server check measures -> check S09 first, then S09-perp if needed.

* S09 cold-start test (history hook disabled, bare 800-row blocks): SR5 +0.0153 +- 0.0063, @10bp +0.0086, IC 0.014,
  turnover 0.0100 (vs warm +0.0259 / 0.0190 / 0.044 / 0.0039). Post-scoring diagnostic: board Sharpe near 0.026 =>
  server calls generate_features (warm); near 0.015 => cold.

## Measured facts (trust these)
1. Index is an integer row number; 31 cumulative period files; latest 50,815 rows; each file adds ~730 rows.
2. **Each period file is an independent RECONSTRUCTION** (dispersion for the same hours differs across files;
   no 1-to-1 ticker match). History inside the latest file carries hindsight. => the ONLY honest evaluation is
   the official per-period protocol (`lab/official.py`): per file NNN train on its rows except last 800 (minus
   127-row embargo), `generate_features(full file)`, predict last 800 rows. Dev = periods 001-029, VAULT = 030-031.
3. Target = xs-z of the 120h return window sum starting 1h ahead (cos 0.936, dispersion corr 0.98). config/derived.json:
   target_lag 1, horizon 120, embargo 127.
4. Hourly Sharpe ~= 0.35 x IC_T (sqrt(N_eff/h), N_eff 14.7). Board leader IC 0.052 -> SR 0.020 fits. 3rd place
   (0.011) needs net IC_T ~0.033.
5. Vendor harness: `predict()` gets 800-row blocks with NO history; `generate_features(full_frame)` hook runs first.
   Strategies S04/S05 locate history BY CONTENT (row hashes) - label-keyed caches would be ROW_MASK risk.
6. Gate emulation (`tools/shuffle_gate.py`, joint row permutation + recomputed target): vendor demo passes on the
   SHARPE statistic (z 7.3, p 0.032 with 30 reps) but not on IC -> judge on Sharpe. Long rolling windows inside a
   model risk failing; same-hour ranks and two-point differences are gate-safe.
7. Board columns decoded: Sharpe | IC | IC std | IC-disp corr | concentration | compression loss | city deg | global deg.
   Admission = global novelty >= 60 deg (|rho| <= 0.5) vs higher-ranked + legacy pot. config/leaderboard.json = snapshot.
8. Quick runner (800 rows) has SE ~0.035: never decide on it (side-agent note, agreed).

## Results so far (official protocol unless stated)
| what | result | verdict |
|---|---|---|
| vendor demo_engineered, runner --full | SR -0.0154, IC 0.0008, city 46.5 deg | negative baseline |
| S04 contemp GBM, runner --gauge-fix | SR -0.0178, IC -0.018, city 29 deg | fails |
| **S04, our official eval** (run 20261006T185130) | **SR -0.0195 [-0.030,-0.011], IC -0.023 t-2.9, 28% periods +, conc 0.76** | significantly ANTI-predictive |
| S04 single-file eval / tilt-off / r04 grid | +0.005 / -0.003 / smoothing helps | INVALID (hindsight protocol) - ignore |
| r05 same-hour screen | F6 rank strongest | single-file => re-check with r06 |
| S01-S03 | built on synthetic only, long rolling windows | not evaluated on real data; gate-risky |

Interpretation (hypothesis, not yet tested): models fitted on in-file history learn a static cross-sectional tilt
(concentration 0.76) that is a hindsight artefact of the reconstruction and REVERSES out of sample. Do NOT just flip
the sign of S04 - that is a fitted-on-failure trap; test the mechanism first (see next steps).

## Interrupted at (re-run these first, ~5-10 min each)
1. `python tools/evaluate.py strategies/s05_ridge_regime.py --mode full` (official protocol) - was mid-run.
2. `python research/r06_official_screen.py` - honest per-period marginal screen (may have finished: check
   reports/r06_official_screen.md timestamp; re-run if missing).

## Next steps (in order)
1. From r06: which constructions have a stable sign ACROSS periods (stab >= 0.65, BH-significant) under the honest
   protocol. Build only on those.
2. Test the tilt hypothesis: S04 with TILT_WINDOW=720 under the official protocol; and a training variant that uses
   only the most recent N rows of each file (less hindsight) - e.g. add MAX_TRAIN_ROWS to S04/S05.
3. Turnover: `python research/r04_turnover.py strategies/<s>.py --halflives 48,168,336,672` (already ported to the
   official protocol, extra warm-up per fold). Raise HISTORY >= smoothing window + 200 in the strategy if hl > 250.
4. For any candidate with shrunk SR > 0 and P(SR>0) >= 0.9: dynamic_check -> shuffle_gate --reps 30 ->
   runner --gauge-fix (city) -> tools/novelty.py -> tools/choose.py --data-tag real-official -> vault (once) ->
   tools/freeze.py promote.
5. If nothing clears the bar: prefer submitting nothing over a negative/fragile signal (every passing submission
   enters the legacy pot). Report honestly.

## Tool status
* Ported to official protocol: evaluate (default), r04_turnover, vault, shuffle_gate (reference = training slice of
  period 029). NOT yet ported: research/r02, r03 (single-file - optimistic), tests/selftest.py (still synthetic-era
  expectations; dynamic_check gained history_hook / no_hook_fallback checks).
* registry.csv: append-only now (concurrency-safe). data tags: `real` = single-file (biased), `real-official` = honest.
