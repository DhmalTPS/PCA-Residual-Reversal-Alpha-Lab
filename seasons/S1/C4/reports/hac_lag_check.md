# HAC-lag sensitivity, S09 (official per-period protocol, dev periods, 5 bp)

The target is a 120 h forward return sampled hourly, so neighbouring IC and PnL observations overlap by up to 119 h.
The screen used `nw_lag_hours = 48`, which understates the long-run variance. The table below is a re-run of
`common/tools/hac_lag_check.py`, which writes nothing to the registry. Only the lag changes.

| NW lag (h) | IC | IC t | SR @5bp | NW s.e. |
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
| As of the S09 evaluation (`--as-of 20261008T050608_s09`) | 0.934 (matches registry row) | **0.951** | 325 |
| Full C4 registry, including 12 later trials | 0.0025 | 0.0018 | 337 |

Twelve later trials widened the spread of Sharpe ratios across comparable trials from about 0.006 to 0.014. That raises the expected-maximum benchmark SR* above S09's 0.0259. **The DSR of 0.93–0.95 holds only for the search as it stood at selection.** Counting every trial in the cycle, S09 does not survive deflation. The per-period evidence does not depend on the lag: 24 of 29 development periods were positive, with a per-period t of about 4.5.
