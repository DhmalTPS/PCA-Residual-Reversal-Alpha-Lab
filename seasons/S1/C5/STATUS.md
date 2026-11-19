# STATUS - S1/C5

(newest first)

Note: IC t and s.e. below use NW lag 48 (< 120 h horizon); expect IC t about 25% lower at lag 120
(see seasons/S1/C4/reports/hac_lag_check.md).

Deviations from PREREGISTRATION.md (recorded 2026-11-19, after the cycle; see its "Deviations" section):
* Rule "a candidate that fails ANY of 1-5 does not get a novelty check" and A1(c) "bars 1-3, 5, 6 stay": A (full
  SR +0.0111), AB (+0.0125) and L240 (+0.0064) were below the bar-1 threshold (SR5 >= 0.0133) and were still
  novelty-checked, as measurements under A1(b). Only Z met the bar.
* "Max 3 novelty checks in C5": 4 were used (A, AB, L240, Z).

* 2026-10-09: folder created; PREREGISTRATION.md and RUNBOOK.md written from C4 evidence (no C5 data yet).

* 2026-10-15 11:05: C5 package installed (identical to C4 data; new city map merged). Active cycle S1/C5. derived.json
  carried over. Pre-registration amendment A1 written (no fresh data -> recency proxy; portfolio of submissions; board
  Sharpe ~= final score).

* 2026-10-15 ~11:45: c5_r01 residual-family grid (16 trials): idiosyncratic reversal strongly positive in the 10 most
  recent periods across the family. Candidates (promoted, all gates PASS incl. gate emulation z 7.9 / 6.4):
  A = L48 K8 hl168 (proxy +0.0254, 10/10 periods +, @10bp +0.0109, full +0.0111, rho S09 0.38, city 48.9 deg / 76 within 60)
  B = L120 K3 hl48 (proxy +0.0279, 8/10, @10bp +0.0170, full +0.0108, rho S09 0.34, city 48.4 deg / 2 within 60); rho(A,B) 0.41.
  Strict v1 bar (full SR >= 0.0133) NOT met; uploaded as measurements under A1. c5_r02 broad recency screen (360 trials):
  0 kept - recent-strong constructions are the occupied reversal ones. Board prediction: board ~ proxy - 0.006 +- 0.01.
  Next: upload A then B with novelty check; record verdicts + board Sharpe; recalibrate proxy.

* 2026-10-15 ~12:30 c5_r03 robustness (pre-declared neighbours, confidence only): A plateau all 7 configs proxy +0.020..+0.036
  (family mean +0.025), period-bootstrap p05 +0.016, 10/10 recent +, cost: @10bp +0.011, @15bp -0.004. B plateau +0.021..+0.029,
  boot sd 0.018 (p05 ~0), @15bp +0.006. S09 calibration: proxy +0.0118 -> board 0.0028 (offset -0.009 or ratio 0.24 - one
  point cannot tell). Board forecast A ~0.016+-0.012 (additive) / ~0.006 (ratio). DECISION: upload A first (most robust),
  then B. Did NOT switch to best neighbour A_L72 (selection bias). Leakage audit: all causal; no unseen local data remains.

* 2026-10-15 ~13:30 novelty check #1: A NOT NOVEL 59.18 deg (rho 0.512; blocker unknown - local overlap with known legs <= 0.38). Discard.
  Built AB = unit(unit(A_s)+unit(B_s)) (strategies/c5_idio_reversal_combo.py): proxy +0.0333 (@10bp +0.0195, @15 +0.0057),
  boot p05 +0.0098, P>0 0.992, vault +0.067, full +0.0125, rho(AB,A)=rho(AB,B)=0.835, rho S09 0.42; all gates PASS, gate
  emulation z 6.2. Promoted: submissions/c5_idio_reversal_combo.py. Novelty: AB clears 60 deg if blocker's cos with B < ~0.33.
  Next: novelty-check AB; if not novel -> B, then triangulate blocker from the two angles.

* 2026-10-15 ~14:15 novelty check #2: AB NOT NOVEL 50.38 deg (rho 0.638). Triangulation (c5_r04_blocker_model.py): blocker
  projection on span{A,B} = 0.339A + 0.421B (64% of its norm); predicted cos(o,X) = 0.339 rho(X,A) + 0.421 rho(X,B);
  reproduces A (0.505 vs 0.512 measured); predicts B at 56 deg -> B NOT checked (saved a check; file moved to ready_not_uploaded/).
  Pre-registered rule (pred cos <= 0.40, rho S09 <= 0.45, boot p05 > 0, max proxy) selected L240 K3 hl48:
  proxy +0.0235 (@10bp +0.0127), boot p05 +0.0054, predicted 70.3 deg, rho S09 0.29, full +0.0064; gates PASS, gate emulation
  z 6.5, vendor de-mean PASS. File: submissions/c5_idio_reversal_change_lag240_n_factors3_smooth_halflife48p0_smooth_window1.py
  A and AB moved to submissions/not_novel/.

* 2026-10-15 ~15:00 novelty check #3: L240 NOT NOVEL 50.78 deg (predicted 70.3) -> blocker model refuted: the pot holds several
  factor-residual reversal signals at different horizons; the family is saturated. City->server calibration on 8 probes: no
  predictive power (city 48-53 deg -> server 38-61 deg) -> map-based screening dropped.
  Geometry of escape: X = w*E + z*Z with E family, Z genuine orthogonal edge: rho_occ ~ 0.64 w -> need w <= 0.70; SR ~ w SR_E + z SR_Z.
  Rule kept: Z must be a GENUINE edge (no noise dilution). c5_r05 (2 pre-declared evals): Z = C4 offline-learned model (weights fit
  on dev 001-029 only; residualised on S06/S07/S08): proxy +0.0333 (10/10), VAULT (clean OOS) +0.012, full +0.0229/@10 +0.0186,
  IC t 5.8, cos legs 0 / S09 0.10, gate_inSR +0.0048, DSR 0.75 -> FIRST candidate to pass admissible_v2.
  X = unit(AB+Z): proxy +0.0457, vault +0.053, full +0.0238, IC t 6.7, rho(X,AB) 0.733 -> predicted occupant rho 0.467.
  Strategy file strategies/c5_orthogonal_learned.py (INCLUDE_AB toggle): reproduces research positions to <= 1e-13 on 6 periods;
  exact-equivalent O(T) rolling stats (constant-window handling) cut predict 59.2 s -> 20.4 s on 49k rows.
  PLAN: novelty-check Z first (passes the bar, most likely novel), then X.

* 2026-10-15 gates on promoted files (submissions/): BOTH READY.
  Z c5_orthogonal_learned.py: precheck PASS, predict 22.9 s / 17.8 s (limit 60), gate projection x50 = 14.8 min,
    shuffle_gate 30 reps: real +0.00475 vs null -0.00788+-0.00129, z 9.8, p 0.032 (rank 1/31) PASS; de-mean PASS.
  X c5_orthogonal_learned_include_abTrue.py: precheck PASS, predict 28.3 s, shuffle_gate: real -0.0126 vs null
    -0.0390+-0.0035, z 7.5, p 0.032 PASS (passes on rank vs null although own in-sample SR is negative); de-mean PASS.
  Next: upload Z with "Check novelty first" (check #4); then X (#5) only if Z verdict is in. Z/X rho ~0.7 -> one star max.

* 2026-10-15 novelty check #4: Z NOT NOVEL - global 58.14 deg (rho 0.528), quality-gate 52.53 deg (rho 0.608).
  FACTS vs INFERENCES written out (to separate measurements from assumptions):
  - FACT: rho at the gate 0.608 > 0.5 -> Z as is cannot enter the quality set while its blocker ranks above it.
  - INFERENCE (unverified): qg < global => blocker in the legacy pot (C4 reading; needs the definitions of the two angles).
  - CORRECTION: AB's/A's vault numbers are NOT clean OOS - A and B were selected by the recency proxy, which includes
    the vault (A1). Z's vault IS clean (weights fit on 001-029), but Z's dev SR is in-sample; honest nested-CV SR from C4 ~0.0075.
  - All blend novelty predictions so far assumed cross-terms ~0 (unmeasured). c5_r06 audits these assumptions.

* 2026-10-15 c5_r06 ASSUMPTION AUDIT (reports/c5_r06_assumption_audit.md; exact server-judged files, all 31 periods):
  T1 net SR full/recent/vault/proxy: AB +0.0125/+0.0281/+0.0665/+0.0333; Z +0.0229(in-sample)/+0.0371/+0.0120/+0.0333;
     X +0.0238/+0.0450/+0.0531/+0.0457; SWD +0.0190/+0.0458/+0.0338/+0.0437; SWC(control) +0.0171/+0.0246/+0.0376/+0.0261.
  T2 blend identity exact (|err| 0) - the only certain part of blend arithmetic.
  T3 no library stand-in (237 fast + slow) reaches |rho| 0.40 with Z (max 0.36) -> Z's blocker is specific; cross-terms
     with AB unknown. Stand-ins near AB have rho(Z) mean +0.07 (max +0.20).
  T4 FAIL: no library statistic ranks the 8 server-measured probes (Spearman |r| <= 0.48, p >= 0.23) -> local
     novelty predictions are UNVALIDATED (same conclusion as the city map).
  T5 AB's correlation with its stand-ins is +0.03..+0.05 higher in high-dispersion hours (slightly against SWD).
  Ratio transfer (assumption: true AB-type blocker behaves like stand-ins): X 0.477 [0.431,0.519], SWD 0.406 [0.363,0.469].
  Timing: AB regime effect t 2.4; AB-vs-Z advantage t 1.4 (not significant).
  strategies/c5_regime_allocation.py (SWD) reproduces research PnL to 1e-17 on 6 periods; gates running.

* 2026-10-15 SWD gate job KILLED (system low on memory) after freeze copied the file (no checklist):
  submissions/c5_regime_allocation.py is NOT READY - rerun freeze/shuffle_gate/dynamic_check/run_vendor/evaluate before any use.
* Vault selection bias quantified (confirmed by an independent check): 16-grid vault mean +0.030 (15/16 > 0, median +0.032) vs selected
  A +0.054 / B +0.053 -> selection uplift ~+0.023; rank corr(recent-dev, vault) +0.16. De-biased vault: AB ~+0.035-0.040,
  X ~+0.034, SWD ~+0.03, Z +0.012 (clean). Fair board expectation for X/SWD ~0.01-0.03 (S09 precedent: proxy 0.013 -> board 0.003).
