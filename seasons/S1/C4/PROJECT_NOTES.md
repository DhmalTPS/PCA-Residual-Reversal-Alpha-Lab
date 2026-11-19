> NOTE (2026-10-08): written for the pre-restructure flat layout. Paths/commands below are historical;
> current layout and commands: <ROOT>/README.md. Old scripts still run via common/tools/run_research.py.

# AlphaNova Biweekly S1C4 - research project

Vendor files (`runner.py`, `predictor.py`, `walkforward.py`, `evaluation.py`, `data_loader.py`,
`city_tools.py`, `demo*.py`, `README.md`, `COMPETITION.md`, `METRICS.md`) are **as shipped - never edit**.
Everything else is my own code.

## Layout
| Path | Role |
|---|---|
| `config/settings.json` | global research settings; `config/derived.json` (written by r01) overrides with measured values |
| `config/leaderboard.json` | public-board snapshot used to calibrate decisions |
| `lab/` | research library (may look ahead only in `lab/labels.py`; **never imported by a strategy**) |
| `research/r0*.py` | one question per script (`# %%` cells); outputs to `reports/` |
| `strategies/s*.py` | each file is a complete, submittable Predictor (single source of truth) |
| `tools/` | gates and pipeline: precheck, dynamic_check, evaluate, shuffle_gate, novelty, choose, vault, freeze |
| `tests/` | selftest + deliberately bad controls each gate must catch |
| `experiments/` | registry.csv (every trial), runs/<id>/ (OOS predictions + metrics), vault_log.csv |
| `pot/` | own passing submissions this season (frozen + predictions) |
| `submit/` | final, checked files with checklist |

## Measured facts (real data, period 031, dev = H039..P029, vault = P030..P031)
* 50,815 hourly rows, integer row index (no timestamps); period files add ~730 rows each.
* Target = xs z-score of the 120h window sum of returns starting one hour ahead (cos 0.936; dispersion corr 0.98).
* Hourly Sharpe ~ 0.35 x IC_T  (= sqrt(N_eff / h), N_eff 14.7, h 120); verified on S04 and on the board leader.
* Overfitting gate: emulated by joint row permutation + recomputed target; vendor demo passes on the Sharpe statistic.
* `predict()` is called on 800-row blocks with no history; history comes from `generate_features()` matched by content.

## Workflow (PowerShell, venv active)
```powershell
$env:PYTHONIOENCODING="utf-8"; $env:COMP6_DISABLE_PIP_INSTALL="1"
python tests/selftest.py                                   # every gate catches its control
python tools/precheck.py strategies/sNN.py                 # static
python tools/dynamic_check.py strategies/sNN.py            # runtime (incl. history hook, row shift)
python tools/evaluate.py strategies/sNN.py --mode full     # official per-period protocol (29 dev periods) + uncertainty
python research/r04_turnover.py strategies/sNN.py          # smoothing x band (paired one-SE rule)
python tools/shuffle_gate.py strategies/sNN.py --reps 30   # gate emulation
python tools/city.py RUN_ID                                # city on DEV periods only (vault-safe)
python tools/novelty.py ; python tools/choose.py           # rho-hat vs pot + greedy selection
python tools/vault.py strategies/sNN.py                    # ONE shot - BEFORE any runner --full/--gauge-fix (they print 030/031)
python tools/freeze.py promote strategies/sNN.py --set K=V # -> submit/ with checklist
```

## Competition rules to keep (summary)
One account; own work; OSI-licensed libraries only; no external data (the synthetic panel in
`lab/synth.py` is a tool test fixture, never used to fit or tune a submission); no private code
sharing; competition data stays private and is deleted after the season; winners must deliver
training + inference code and an environment description (this file + `requirements-lab.txt`).
