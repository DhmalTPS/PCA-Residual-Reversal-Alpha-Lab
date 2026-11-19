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
