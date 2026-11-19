# Submission checklist - s09_pca_idio_reversal

Source: `strategies/s09_pca_idio_reversal.py`  overrides baked: `{}`  sha: `42e5f14f13d6ed58`  created: 2026-10-08T05:09:52+00:00

Automated (re-run on this exact file):
- [x] static gate (tools/precheck.py)
- [x] runtime gate (tools/dynamic_check.py)

Evidence (fill from experiments/registry.csv and reports/):
- [ ] full walk-forward: NW 90% CI of net Sharpe @10bp excludes 0
- [ ] deflated Sharpe prob > 0.5 with the honest trial count
- [ ] shuffle gate p < 0.05 (tools/shuffle_gate.py)
- [ ] |rho| 90% upper bound <= 0.35 vs pot/ and vs the other submissions of this cycle (tools/choose.py)
- [ ] `python runner.py s09_pca_idio_reversal.py` (official quick + de-mean check) PASS
- [ ] `python runner.py s09_pca_idio_reversal.py --gauge-fix`: city distance recorded
- [ ] vault run once, same sign as dev, inside the dev CI (tools/vault.py)
- [ ] submitted >= 24 h before the deadline

Rules compliance:
- [ ] own original work; only OSI-licensed libraries (numpy/pandas/scikit-learn/lightgbm: BSD/MIT)
- [ ] no external data; no hand labelling
- [ ] code not shared privately with anyone
