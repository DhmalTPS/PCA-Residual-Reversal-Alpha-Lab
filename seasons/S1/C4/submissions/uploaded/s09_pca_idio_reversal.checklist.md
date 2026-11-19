# Submission checklist - s09_pca_idio_reversal

Source: `strategies/s09_pca_idio_reversal.py`  overrides baked: `{}`  sha: `42e5f14f13d6ed58`  created: 2026-10-08T05:09:52+00:00

Automated (re-run on this exact file):
- [x] static gate (tools/precheck.py)
- [x] runtime gate (tools/dynamic_check.py)

Evidence (fill from experiments/registry.csv and reports/):
- [ ] full walk-forward: NW 90% CI of net Sharpe @10bp excludes 0
- [ ] deflated Sharpe prob > 0.5 with the honest trial count (0.93 as first logged at N=325 used the candidate's own s.e.
  as the trial spread; ~0 once the PERP_S08 sibling is counted, N=326. NOT met; see reports/hac_lag_check.md)
- [x] shuffle gate p < 0.05 (tools/shuffle_gate.py): real -0.0116 vs null -0.0246 +- 0.0037, z 3.5, PASS
- [ ] |rho| 90% upper bound <= 0.35 vs pot/ and vs the other submissions of this cycle (tools/choose.py)
- [ ] `python runner.py s09_pca_idio_reversal.py` (official quick + de-mean check) PASS
- [ ] `python runner.py s09_pca_idio_reversal.py --gauge-fix`: city distance recorded
- [ ] vault run once, same sign as dev, inside the dev CI (tools/vault.py) (NOT done before upload: see STATUS.md;
  run after upload: SR -0.0103, opposite sign)
- [x] submitted >= 24 h before the deadline (2026-10-08; close 2026-10-15)

Rules compliance:
- [x] own original work; only OSI-licensed libraries (numpy/pandas/scikit-learn/lightgbm: BSD/MIT)
- [x] no external data; no hand labelling
- [ ] code not shared privately with anyone
