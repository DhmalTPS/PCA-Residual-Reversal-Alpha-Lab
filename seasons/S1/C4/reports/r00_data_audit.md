# R00 data audit (real, period 031)

> Note: STATUS.md and PROJECT_NOTES.md record 50,815 rows for the latest file; the 49,351 below comes from this
> earlier run and was not reconciled.

## Structure
- rows 49351, tickers 20, features ['Feature.1', 'Feature.2', 'Feature.3', 'Feature.4', 'Feature.5', 'Feature.6']
- index 0 .. 49350; most common steps: n/a
- periods on disk: ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031'] (cumulative: only the latest is used)
- target warm-up: first 3462 rows all-zero; all-zero rows in total 3462
- target range [-3.30, 3.79], clipped share 0.0000
- dispersion file: present (FORWARD-looking: diagnostics only)
- mean corr of same-named ticker across the last two periods: +0.423 (~1 would mean NO rotation)

## Features
level_ac1 near 1 = slow state; cos(dF, X) far from 0 = feature changes co-move with returns (return-like).
| feature | nan_frac | level_ac1 | diff_ac1 | xs_sd_median | scale_ratio_max_min | cos(dF, X) |
|---|---|---|---|---|---|---|
| Feature.1 | +0.0000 | +0.9687 | -0.0346 | +0.0297 | +2.2277 | +0.5231 |
| Feature.2 | +0.0000 | +0.9687 | -0.0343 | +0.0297 | +2.2266 | -0.0263 |
| Feature.3 | +0.0000 | +0.9732 | +0.0934 | +0.0303 | +2.4097 | +0.2943 |
| Feature.4 | +0.0000 | +0.9660 | +0.0302 | +0.0293 | +2.0602 | +0.2358 |
| Feature.5 | +0.0000 | +0.1246 | -0.4281 | +0.5910 | +15.3811 | +0.0870 |
| Feature.6 | +0.0000 | +0.9685 | -0.0000 | +0.0001 | +2.3843 | -0.0083 |

## Returns geometry
- hourly sd (median asset) 0.00800; mean |X| 0.00477
- mean pairwise corr of de-meaned returns -0.052; participation-ratio breadth N_eff = 14.7 of 20
- top eigenvalue share 0.124
- fundamental-law scale: IC 0.01 x sqrt(N_eff) = gross hourly SR ~ 0.038