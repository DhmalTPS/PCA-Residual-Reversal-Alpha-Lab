# PCA-Residual Reversal Alpha Lab

A cross-sectional signal (reversal of the PCA-idiosyncratic 120 h change across 20 assets) and the evaluation lab
built to test it honestly: an embargoed per-period out-of-sample protocol, a held-out vault, a permutation null for
the organiser's overfitting gate, deflated Sharpe over every logged trial, and a self-test suite of deliberately
broken controls. The signal passed the organiser's gates; its edge did not survive the most recent data.

## Context

- AlphaNova Biweekly Competition, "Cross-Sectional Signal Forecasting", Season 1, cycles C4 and C5
  (October 2026). Solo, live-scored, run by a fund.
- **Authorship:** the research design, evaluation protocol and final selection are mine. I used AI coding assistants for parts of the implementation and for adversarial review.
- Task: at each hour, output a cross-sectionally de-meaned signal over J = 20 anonymised assets. The score is the
  Sharpe of `r(i) = <P(i-1), X(i)> - c * ||P(i-1) - P(i-2)||_1` with c = 5 bp, where X is a proprietary,
  constructed return panel. The target is undisclosed, and the six features are anonymised.
- Entry: up to 20 submissions per cycle. Each passes code checks and a shuffled-label overfitting gate. Prizes go
  to the top 3 of a "quality set": a signal joins it only if |corr| <= 0.5 (>= 60 deg) against every higher-ranked
  signal and the season's legacy pot of earlier entries.
- All Sharpe ratios below are per hour, as the organiser reports them. No annualised figure is claimed.

## Problem

The obvious approach is to fit a model on the latest shipped file and validate on its tail. It gives misleading
answers here, for two reasons I measured early on:

1. **Each period file is an independent reconstruction with hindsight.** The same hours differ across files, and
   the oldest rows of every file show the opposite sign to the scored rows (`seasons/S1/C4/reports/r07_recency.md`).
   A contemporaneous GBM looked positive under single-file evaluation (+0.005). Under the per-period protocol it
   came out significantly anti-predictive: SR -0.0195 [-0.030, -0.011], IC t -2.9, positive in 28% of periods.
2. **The target is hidden.** I identified it as the cross-sectional z-score of the 120 h forward return sum, starting
   1 h ahead (cos 0.936 overall, exact on the scored rows; `seasons/S1/C4/config/derived.json`). That fixes the
   horizon and therefore the embargo.

## Approach

### Evaluation lab (`common/lab`, `common/tools`, `common/tests`)

| Component | What it does | Where |
|---|---|---|
| Per-period OOS protocol | For each period file: train on all rows except the last 800, minus a 127 h embargo (120 h horizon + 1 h lag + 6 h buffer); build features on the full file; score only the last 800 rows. Dev = periods 001-029; vault = 030-031, opened once per candidate. | `lab/official.py`, `lab/paths.py` |
| Honest harness | Net SR at 5 and 10 bp, Newey-West s.e. (default lag 48; see the HAC correction below), 90% stationary-bootstrap CI, IC with NW t, share of positive periods, cosine overlap with occupied signal directions, in-sample gate SR, recent-period and vault SR | `lab/honest.py`, `lab/metrics.py` |
| Deflated Sharpe | Every `evaluate()` call is appended to `experiments/registry.csv`. DSR uses that trial count and the cross-trial SR dispersion. | `lab/registry.py`, `lab/metrics.py` |
| Permutation null | Emulates the organiser's gate: rows of features and returns are permuted jointly, the target is recomputed from the permuted returns, the model is retrained, and the result is judged on net Sharpe | `lab/nulls.py`, `tools/shuffle_gate.py` |
| Pre-registered bar | Fixed before a server novelty check is spent: SR5 >= 0.0133, CI lower bound > 0, SR10 > 0.008, >= 65% positive periods, DSR >= 0.5, max \|cos\| with occupied legs <= 0.35. `admissible_v2` adds recent SR > 0 and vault SR >= 0. | `lab/honest.py`, `seasons/S1/C5/PREREGISTRATION.md` |
| Self-test | Every gate must pass the clean strategies and catch each control: pure noise (`c00`), next-row leakage (`c01`), a row-position on/off mask (`c02`), logic outside the class (`c03`). 21/21 expectations met. | `tests/selftest.py`, `tests/controls/` |
| Static / runtime gates | Static scan for look-ahead and code-structure violations. Runtime checks: causality (outputs for hours <= t identical with or without later rows), row-shift invariance, flat periods, timing. | `tools/precheck.py`, `tools/dynamic_check.py` |

### Signal: S09, PCA-idiosyncratic reversal (`seasons/S1/C4/strategies/s09_pca_idio_reversal.py`)

1. Estimate the 20 x 20 covariance of hourly changes of Feature.1 over a trailing 720 h window (exact windowed sums).
2. Take the top K = 8 eigenvectors as the common-factor subspace.
3. Project that subspace out of each asset's 120 h change. The position is the negative of the residual
   (idiosyncratic reversal).
4. Unit-normalise each hour, apply exact O(T) finite exponential smoothing (half-life 672 h) to control 5 bp
   turnover, and de-mean.

Nothing is fitted. History is recovered by matching row content, because the server calls `predict()` on
800-row blocks. A reusable copy is in `common/strategy_templates/t_factor_residual_reversal__from_S1C4_s09.py`.

## Results

C4 numbers come from `seasons/S1/C4/STATUS.md` and `CYCLE.md`. All are net, per hour, under the per-period protocol
unless stated otherwise.

| Item | Result |
|---|---|
| S09 dev net SR @5 bp | **+0.0259**, 90% CI [+0.0166, +0.0352] |
| S09 dev net SR @10 bp (stress) | +0.0190 |
| IC t / periods positive / DSR | **3.6 / 24 of 29 (83%) / 0.95 at selection** (N = 325 trials logged at evaluation). Counting all 337 trials in the C4 registry, the DSR is about 0.002, because later trials widened the spread of Sharpe ratios across trials. See `seasons/S1/C4/reports/hac_lag_check.md` |
| HAC correction | The original screen used a Newey-West lag of 48 h, which is shorter than the 120 h overlap of the target, and gave IC t 4.8. With a lag of 120 h (127 h: 3.5) the IC t is 3.6, and Sharpe is unchanged. See `seasons/S1/C4/reports/hac_lag_check.md` and `common/tools/hac_lag_check.py`. |
| Stability | half-samples 001-014 / 015-029: +0.025 / +0.027; smoothing half-life 336-1344 h: +0.024 to +0.027 |
| Permutation null (in-sample) | real -0.0116 vs null -0.0246 +- 0.0037, z 3.5: pass |
| Organiser gates | passed the real overfitting gate; novel at 61.25 deg (rho 0.481) against the 60 deg bar |
| Vault 030-031 (**run after upload**) | SR -0.0103 +- 0.0205, IC -0.087 (t -2.5) |
| Live board | not in the quality set, **no prize** |
| Baselines | vendor demo -0.0154; contemporaneous GBM -0.0195 (IC t -2.9) |

**Reading.** The edge was real on the shipped history: it was stable across halves and smoothing choices, and it
survived deflation for 325 trials and the organiser's gate. It decayed in the most recent data, and the board
scores a hidden band newer than the shipped files. By sub-period, the dev SR was +0.029 on periods 001-010, +0.034 on
011-020 and +0.016 on 021-029, and the vault was negative. The raw signal stayed positive in the vault (IC +0.032),
so the 4-week smoothing lagged the regime. The vault should have vetoed the upload. It was not run first, and the
log records this as a process error (`C4/STATUS.md`). From then on it is a hard rule: vault every candidate before
any server check.

The upload-time server Sharpe (0.0241) was measured on the shipped history. It is not a board score and is not
used as a result.

**C5 (follow-up, no entry).** I wrote a pre-registration before any C5 data arrived (`C5/PREREGISTRATION.md`) and
amended it with a dated entry. Residual-family grids, a 360-trial recency screen and a blocker-triangulation model
produced candidates that passed every local gate. All four server novelty checks came back not novel (50-59 deg):
the factor-residual reversal family was saturated in the pot. I also measured my own vault selection bias: the
16-config grid mean was +0.030, against +0.054 for the selected configs (`C5/STATUS.md`,
`C5/reports/c5_r06_assumption_audit.md`).

## Repository layout

```
common/                      reusable across cycles
  lab/                       official.py, honest.py, metrics.py (NW, bootstrap, PSR/DSR, BH), nulls.py,
                             registry.py, backtest.py, gauge.py, paths.py, synth.py; labels.py is research-only
                             (look-ahead) and never imported by a strategy
  tools/                     precheck, dynamic_check, evaluate, shuffle_gate, vault, city, novelty, choose,
                             freeze, run_research, run_vendor, new_cycle, set_cycle, make_synth
  tests/                     selftest.py + controls/ (c00 noise, c01 leaky, c02 rowmask, c03 top-level)
  research_templates/        cycle-aware research scripts (target identification, official screens, turnover)
  strategy_templates/        compliant patterns extracted from S04/S06/S07/S09
  config/defaults.json
seasons/S1/
  occupied_directions.py     signal directions known to be occupied (from server novelty checks)
  C4/  STATUS.md (full decision log), CYCLE.md (outcome), PROJECT_NOTES.md
       strategies/ S01-S09   research/ r00-r16 + research/agents/ (family searches incl. peer_factor_final.py)
       reports/              screens, gate/vault/city JSONs     experiments/ registry.csv, vault_log.csv
       config/               settings.json, derived.json (measured target horizon, lag, embargo)
       submissions/uploaded/ the exact file submitted (S09) + checklist
  C5/  PREREGISTRATION.md, RUNBOOK.md, STATUS.md, strategies/, research/, reports/, experiments/
JOURNAL.md                   one line per cycle
```

## How to run

The competition data and the organiser's package (runner, `predictor.py`, period files, signal-city map) are
proprietary and are not included. Without them, the tools fall back to a synthetic panel, and `lab/_stub/predictor.py`
stands in for the base class.

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-lab.txt

# tool tests on a synthetic panel (never used for fitting)
python common/tools/make_synth.py
export ALPHANOVA_CYCLE=S1/C4                          # PowerShell: $env:ALPHANOVA_CYCLE="S1/C4"
python common/tests/selftest.py

# with the package unpacked into seasons/S1/C4/vendor/ (data/train/*.parquet)
python common/research_templates/r01_target_identification.py --write   # -> config/derived.json
python common/tools/evaluate.py seasons/S1/C4/strategies/s09_pca_idio_reversal.py --mode full
python common/tools/shuffle_gate.py seasons/S1/C4/strategies/s09_pca_idio_reversal.py --reps 30
python common/tools/vault.py seasons/S1/C4/strategies/s09_pca_idio_reversal.py   # one shot, before any server check
```

On Windows the vendor runner needs `PYTHONIOENCODING=utf-8` and `COMP6_DISABLE_PIP_INSTALL=1`
(`common/tools/run_vendor.py` sets both). Every `evaluate()` call is logged as a trial, and that log is what the
deflated Sharpe uses.

## Notes and limitations

- The outcome is a negative result. The signal was accepted, but it did not make the quality set and won no prize.
  The work worth looking at is the evaluation discipline and the diagnosis of why the edge faded.
- The vault for S09 was opened after the upload decision, so the dev metrics alone are not clean held-out evidence.
- The gate emulation was calibrated on only one or two server data points. It is too strict for reversal signals:
  a pot signal that local emulation failed still passed the real gate.
- Search breadth: S09 came out of a multi-family search, and the half-life was chosen from a sweep. The trial count
  behind the DSR includes every logged screen. The S09 docstring quotes slightly different figures from an earlier
  registry state (IC t 6.0, DSR 0.94, N ~320). The table above uses the `STATUS.md` values.
- Asset and feature identities are unknown, and the return panel is constructed by the organiser. Nothing here
  claims to generalise to real market returns.
- S01-S03 were written against the synthetic panel and are kept only as history.
