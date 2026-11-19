# common/ - reusable across every season and cycle

* `lab/` - library. `paths.py` resolves everything from the ACTIVE cycle (`<ROOT>/ACTIVE_CYCLE` or env `ALPHANOVA_CYCLE`).
  Key modules: `official.py` (honest per-period protocol), `honest.py` (shared candidate harness + pre-registered bar),
  `metrics.py` (NW s.e., bootstrap, PSR/DSR, shrinkage), `nulls.py` (gate emulation), `gauge.py`, `registry.py`,
  `io.py`, `walkforward.py` (single-file protocol - biased on shipped data, kept for reference), `labels.py` (look-ahead,
  research only), `synth.py`.
  `honest.py` reads the season's occupied directions from `seasons/<S>/occupied_directions.py` (new seasons start empty).
* `tools/` - run from the root, e.g. `python common\tools\evaluate.py seasons\S1\C4\strategies\s09_pca_idio_reversal.py`.
* `tests/selftest.py` - every gate must PASS clean strategies and CATCH each bad control.
* `research_templates/` - copies of reusable research scripts; they work on the active cycle from any folder.
  Copy into `<cycle>/research/` to start a cycle's own research (keep templates pristine).
* `strategy_templates/` - copies of proven, compliant patterns (history-by-content hook, exact O(T) smoother,
  fixed or learned). Copy into `<cycle>/strategies/` and adapt; always re-run every gate.
* `config/defaults.json` - defaults; each cycle overrides in `<cycle>/config/settings.json`.
