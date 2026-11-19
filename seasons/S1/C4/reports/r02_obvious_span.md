# R02 obvious span (real)

IC_T: mean per-hour IC vs the TARGET (NW t). grossSR_X / netSR_X: PnL on the returns panel (net uses 12h smoothing, official cost).
BH_q10: survives Benjamini-Hochberg at q=10% across this family.
**Signals with |IC_t| small but |grossSR_X| material = what the target neutralises. Orthogonalise candidates against them.**

| signal | IC_T | IC_t | p | grossSR_X | netSR_X_smoothed | SR_se | BH_q10 |
|---|---|---|---|---|---|---|---|
| Feature.6 level | -0.0304 | -5.1840 | +0.0000 | -0.0166 | -0.0302 | +0.0045 | True |
| Feature.5 chg24 | +0.0100 | +3.2561 | +0.0011 | +0.0038 | -0.0132 | +0.0054 | True |
| returns trailing 6h (momentum) | +0.0067 | +1.7258 | +0.0844 | -0.0150 | +0.0004 | +0.0049 | False |
| returns trailing 1h (momentum) | +0.0024 | +1.4700 | +0.1415 | -0.0412 | -0.0407 | +0.0049 | False |
| returns trailing 96h (momentum) | +0.0113 | +1.1649 | +0.2441 | +0.0149 | +0.0144 | +0.0046 | False |
| Feature.4 level | -0.0048 | -0.6927 | +0.4885 | +0.0118 | +0.0034 | +0.0048 | False |
| Feature.2 level | -0.0048 | -0.6879 | +0.4915 | +0.0146 | +0.0042 | +0.0048 | False |
| returns trailing 24h (momentum) | +0.0049 | +0.6804 | +0.4963 | +0.0123 | +0.0107 | +0.0048 | False |
| Feature.3 chg24 | +0.0032 | +0.6782 | +0.4976 | +0.0113 | +0.0015 | +0.0050 | False |
| Feature.1 level | -0.0045 | -0.6457 | +0.5185 | +0.0077 | +0.0040 | +0.0049 | False |
| Feature.3 level | -0.0044 | -0.6315 | +0.5277 | +0.0127 | +0.0046 | +0.0049 | False |
| Feature.1 chg24 | +0.0030 | +0.6294 | +0.5291 | +0.0049 | +0.0010 | +0.0049 | False |
| Feature.4 chg24 | +0.0027 | +0.5742 | +0.5658 | +0.0120 | +0.0011 | +0.0049 | False |
| Feature.2 chg24 | +0.0026 | +0.5422 | +0.5877 | +0.0157 | +0.0016 | +0.0049 | False |
| Feature.6 chg24 | +0.0013 | +0.3997 | +0.6894 | -0.0067 | -0.0331 | +0.0043 | False |
| Feature.5 level | -0.0011 | -0.2607 | +0.7943 | +0.0193 | -0.0013 | +0.0057 | False |

## Strategy overlap with the obvious set (mean per-hour cosine)
| strategy | top_overlaps |
|---|---|
| s04_xs_contemp_gbm | Feature.6 level -0.64; Feature.6 chg24 +0.09; returns trailing 96h (momentum) -0.06 |
| s04_xs_contemp_gbm | Feature.6 level -0.36; Feature.6 chg24 +0.11; returns trailing 96h (momentum) -0.08 |