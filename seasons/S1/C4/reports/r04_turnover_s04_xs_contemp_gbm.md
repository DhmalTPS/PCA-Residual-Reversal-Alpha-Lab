# R04 turnover: s04_xs_contemp_gbm (real)

Grid over smoothing half-life (finite kernel, window = 4 x half-life) and no-trade band (fraction of the typical |position|).
Stress cost 10 bp used for the choice; one-SE rule (paired-difference s.e.) toward MORE smoothing.

| halflife | window | band | SR_gross | SR_5bp | SR_10bp | se | turnover | se_diff_vs_best |
|---|---|---|---|---|---|---|---|---|
| +336.0000 | +1344.0000 | +0.0000 | +0.0086 | +0.0075 | +0.0063 | +0.0068 | +0.0073 | +0.0000 |
| +168.0000 | +672.0000 | +0.0000 | +0.0100 | +0.0080 | +0.0059 | +0.0067 | +0.0135 | +0.0017 |
| +96.0000 | +384.0000 | +0.0000 | +0.0100 | +0.0067 | +0.0035 | +0.0066 | +0.0225 | +0.0030 |
| +48.0000 | +192.0000 | +0.0000 | +0.0108 | +0.0052 | -0.0004 | +0.0066 | +0.0423 | +0.0044 |
| +24.0000 | +96.0000 | +0.0000 | +0.0128 | +0.0036 | -0.0057 | +0.0068 | +0.0781 | +0.0052 |

**Best @10bp**: half-life 336.0, band 0.0 -> SR +0.0063 (se 0.0068)
**Chosen (one-SE, most conservative)**: half-life 336.0, band 0.0 -> SR@10bp +0.0063, SR@5bp +0.0075
**WARNING: choice sits on the grid edge (half-life, band) - extend the grid before trusting it.**
Re-evaluate with: `python tools/evaluate.py ../strategies/s04_xs_contemp_gbm.py --set SMOOTH_HALFLIFE=336.0 --set SMOOTH_WINDOW=1344`
5 grid points were logged as trials.