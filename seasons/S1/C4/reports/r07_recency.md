# R07 - within-file recency drift (real data, 2026-10-07)

Mean per-hour IC vs target, averaged over dev period files 001, 003, ..., 029 (15 files);
"pos" = share of files with IC > 0. Segments are row offsets from each file's END.

| construction | old train rows (-20k..-10k) | recent train rows (-5k..-927) | validation (last 800) |
|---|---|---|---|
| rank Feature.1 | +0.0043 (0.60) | -0.0236 (0.27) | -0.0396 (0.33) |
| rank Feature.4 | +0.0046 (0.60) | -0.0236 (0.27) | -0.0397 (0.33) |
| rank Feature.6 | -0.0237 (0.20) | +0.0013 (0.67) | +0.0177 (0.60) |
| rank Feature.5 | -0.0009 (0.27) | -0.0088 (0.27) | -0.0147 (0.33) |
| rank zF1 - zF6 | +0.0164 (0.67) | -0.0166 (0.27) | -0.0395 (0.27) |
| rank d120 Feature.1 | +0.0050 (0.73) | -0.0208 (0.33) | -0.0314 (0.27) |

Reading: the oldest history inside every file has the OPPOSITE sign to the scored rows; recent history has
the SAME sign, weaker. Consistent with hindsight in each file's reconstruction (e.g. universe chosen as of
the file date). Consequence: train on the most recent rows only (`MAX_TRAIN_ROWS` in S04/S05), and treat any
result from full-history fitting or single-file evaluation as biased.
