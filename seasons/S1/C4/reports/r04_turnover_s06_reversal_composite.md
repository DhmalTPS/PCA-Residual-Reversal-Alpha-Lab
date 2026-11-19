# R04 turnover: s06_reversal_composite (real-official)

Grid over smoothing half-life (finite kernel, window = 4 x half-life) and no-trade band (fraction of the typical |position|).
Stress cost 10 bp used for the choice; one-SE rule (paired-difference s.e.) toward MORE smoothing.

| halflife | window | band | SR_gross | SR_5bp | SR_10bp | se | turnover | se_diff_vs_best |
|---|---|---|---|---|---|---|---|---|
| +672.0000 | +2688.0000 | +0.0000 | +0.0232 | +0.0203 | +0.0175 | +0.0065 | +0.0041 | +0.0000 |
| +336.0000 | +1344.0000 | +0.0000 | +0.0238 | +0.0197 | +0.0156 | +0.0064 | +0.0081 | +0.0025 |
| +168.0000 | +672.0000 | +0.0000 | +0.0248 | +0.0192 | +0.0136 | +0.0061 | +0.0159 | +0.0044 |
| +96.0000 | +384.0000 | +0.0000 | +0.0273 | +0.0200 | +0.0127 | +0.0061 | +0.0272 | +0.0055 |
| +48.0000 | +192.0000 | +0.0000 | +0.0307 | +0.0210 | +0.0112 | +0.0062 | +0.0514 | +0.0065 |

**Best @10bp**: half-life 672.0, band 0.0 -> SR +0.0175 (se 0.0065)
**Chosen (one-SE, most conservative)**: half-life 672.0, band 0.0 -> SR@10bp +0.0175, SR@5bp +0.0203
**WARNING: choice sits on the grid edge (half-life, band) - extend the grid before trusting it.**
Re-evaluate with: `python tools/evaluate.py ../strategies/s06_reversal_composite.py --set SMOOTH_HALFLIFE=672.0 --set SMOOTH_WINDOW=2688`
5 grid points were logged as trials.