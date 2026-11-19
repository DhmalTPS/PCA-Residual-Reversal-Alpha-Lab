# HAC-lag sensitivity, S09 (official per-period protocol, dev periods, 5 bp)

The target is a 120 h forward return sampled hourly, so neighbouring IC and PnL observations overlap by up to 119 h.
The screen used `nw_lag_hours = 48`, which understates the long-run variance. The table below is a re-run of
`common/tools/hac_lag_check.py`, which writes nothing to the registry. Only the lag changes.

| NW lag (h) | IC | IC t | SR @5bp | SR NW s.e. |
|---|---|---|---|---|
| 48 (original) | 0.0438 | 4.76 | 0.0259 | 0.0059 |
| 120 (= horizon) | 0.0438 | **3.59** | 0.0259 | 0.0057 |
| 127 (= embargo) | 0.0438 | 3.55 | 0.0259 | 0.0057 |

The IC t falls from 4.8 to 3.6 once the lag covers the overlap. That is still well above conventional thresholds.
`defaults.json` is unchanged, so that the logged trials stay comparable. New screens should use a lag of at least the target horizon.

## Deflated Sharpe (DSR) sensitivity

The DSR depends on N and on the cross-trial spread of the Sharpe ratios of comparable ("full") trials, so the result depends on how much of the registry is counted.

| Registry used | NW lag 48 | NW lag 120 | N |
|---|---|---|---|
| As first logged (`--as-of 20261008T050608_s09`; sibling variant excluded, spread fallback) | 0.934 (matches registry row) | **0.951** | 325 |
| Including the PERP_S08 sibling logged 18 s later (same selection step) | ~0 | ~0 | 326 |
| Full C4 registry | 0.0025 | 0.0018 | 337 |

At S09's evaluation the registry held only 8 comparable (`full`) trials, fewer than the 10 the spread estimator needs (`common/lab/registry.py:33-43`), so the DSR fell back to S09's own s.e. (about 0.006) as the trial spread. That fallback is what produced 0.93–0.95; it is not a cross-trial spread. S09 was chosen over its PERP_S08=True sibling after both were evaluated, and the sibling, logged 18 s later, is the 10th comparable trial. Counting it (N = 326) makes the real cross-trial spread (0.014) available, which raises the expected-maximum benchmark SR* to about 0.042, above S09's 0.0259, so **the DSR is about 0. S09 does not survive deflation**, and the 0.93–0.95 should not be read as evidence. The evidence that remains is the IC t (3.6 at lag 120, 4.76 at lag 48) and the per-period record. The per-period evidence does not depend on the lag: 24 of 29 development periods were positive, with a per-period t of about 4.5.
