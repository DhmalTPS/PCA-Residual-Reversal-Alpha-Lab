# C5 r06 assumption audit

## T1 - net SR @5bp per candidate (NW s.e., lag 48), exact server-judged files
| cand | full dev 001-029 | recent dev 022-029 | vault 030-031 | proxy (10) | provenance caveat |
|---|---|---|---|---|---|
| A | +0.0072 (0.0060) | +0.0197 (0.0112) | +0.0559 (0.0196) | +0.0244 (0.0099) | selected on proxy (incl. vault) from 16-grid |
| AB | +0.0125 (0.0064) | +0.0281 (0.0128) | +0.0665 (0.0195) | +0.0333 (0.0113) | built from A,B selected on proxy (incl. vault) |
| L240 | +0.0069 (0.0062) | +0.0221 (0.0126) | +0.0340 (0.0217) | +0.0235 (0.0111) | chosen by blocker model + proxy rule |
| Z | +0.0229 (0.0063) | +0.0371 (0.0122) | +0.0120 (0.0221) | +0.0333 (0.0109) | weights fit on dev 001-029 -> dev in-sample; vault clean |
| X | +0.0238 (0.0063) | +0.0450 (0.0123) | +0.0531 (0.0193) | +0.0457 (0.0108) | AB + Z: both caveats |
| SWD | +0.0190 (0.0063) | +0.0458 (0.0122) | +0.0338 (0.0197) | +0.0437 (0.0108) | pre-declared, first look |
| SWC | +0.0171 (0.0062) | +0.0246 (0.0115) | +0.0376 (0.0202) | +0.0261 (0.0102) | CONTROL only |
| S06 | +0.0192 (0.0063) | +0.0104 (0.0127) | +0.0062 (0.0184) | +0.0097 (0.0111) | C4 probe |
| S07 | +0.0114 (0.0065) | +0.0090 (0.0125) | +0.0186 (0.0193) | +0.0103 (0.0109) | C4 probe |
| S08 | +0.0221 (0.0062) | +0.0270 (0.0120) | +0.0080 (0.0223) | +0.0241 (0.0108) | C4 probe |
| S09 | +0.0259 (0.0060) | +0.0170 (0.0110) | -0.0097 (0.0206) | +0.0131 (0.0099) | our pot entry |

## Probe x probe time-averaged cross-sectional correlation (rho_hat, scored rows of all 31 periods)
| | A | AB | L240 | Z | X | SWD | SWC | S06 | S07 | S08 | S09 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | +1.00 | +0.83 | +0.35 | +0.05 | +0.59 | +0.50 | +0.49 | +0.14 | -0.02 | +0.19 | +0.38 |
| AB | +0.83 | +1.00 | +0.52 | +0.09 | +0.73 | +0.61 | +0.61 | +0.16 | -0.02 | +0.32 | +0.42 |
| L240 | +0.35 | +0.52 | +1.00 | +0.07 | +0.40 | +0.36 | +0.32 | +0.16 | -0.01 | +0.33 | +0.29 |
| Z | +0.05 | +0.09 | +0.07 | +1.00 | +0.73 | +0.68 | +0.61 | -0.00 | -0.00 | +0.00 | +0.11 |
| X | +0.59 | +0.73 | +0.40 | +0.73 | +1.00 | +0.88 | +0.82 | +0.11 | -0.01 | +0.22 | +0.36 |
| SWD | +0.50 | +0.61 | +0.36 | +0.68 | +0.88 | +1.00 | +0.72 | +0.10 | -0.01 | +0.19 | +0.32 |
| SWC | +0.49 | +0.61 | +0.32 | +0.61 | +0.82 | +0.72 | +1.00 | +0.08 | -0.01 | +0.17 | +0.29 |
| S06 | +0.14 | +0.16 | +0.16 | -0.00 | +0.11 | +0.10 | +0.08 | +1.00 | -0.01 | +0.45 | +0.19 |
| S07 | -0.02 | -0.02 | -0.01 | -0.00 | -0.01 | -0.01 | -0.01 | -0.01 | +1.00 | -0.01 | +0.03 |
| S08 | +0.19 | +0.32 | +0.33 | +0.00 | +0.22 | +0.19 | +0.17 | +0.45 | -0.01 | +1.00 | +0.27 |
| S09 | +0.38 | +0.42 | +0.29 | +0.11 | +0.36 | +0.32 | +0.29 | +0.19 | +0.03 | +0.27 | +1.00 |

## T2 - linear blend prediction vs actual over the library (X and SWD)
X: hourly-identity prediction error over 237 stand-ins: mean -0.0000, max |err| 0.0000
SWD: hourly-identity prediction error over 237 stand-ins: mean -0.0000, max |err| 0.0000

## T3 - cross-terms: stand-ins near one component, their correlation with the other
no stand-in with |rho(Z,l)| >= 0.40
65 stand-ins with |rho(AB,l)| >= 0.40: signed rho(Z,l) mean +0.070, range [-0.011, +0.197]
   idio_F1_L120_K3@48       rho(AB) +0.830  rho(Z) +0.096  rho(X) +0.624  rho(SWD) +0.528  rho(SWC) +0.516
   idio_F2_L120_K3@48       rho(AB) +0.828  rho(Z) +0.096  rho(X) +0.622  rho(SWD) +0.527  rho(SWC) +0.515
   idio_F1_L120_K5@48       rho(AB) +0.789  rho(Z) +0.084  rho(X) +0.587  rho(SWD) +0.500  rho(SWC) +0.483
   idio_F3_L120_K3@48       rho(AB) +0.784  rho(Z) +0.095  rho(X) +0.591  rho(SWD) +0.500  rho(SWC) +0.490
   idio_F4_L120_K3@48       rho(AB) +0.767  rho(Z) +0.095  rho(X) +0.581  rho(SWD) +0.495  rho(SWC) +0.477
   idio_F1_L120_K8@48       rho(AB) +0.763  rho(Z) +0.085  rho(X) +0.570  rho(SWD) +0.488  rho(SWC) +0.466
   idio_F1_L48_K8@336       rho(AB) +0.758  rho(Z) +0.048  rho(X) +0.543  rho(SWD) +0.465  rho(SWC) +0.445
   idio_F1_L48_K8@48        rho(AB) +0.730  rho(Z) +0.037  rho(X) +0.516  rho(SWD) +0.432  rho(SWC) +0.425

## T4 - does any library statistic rank the 8 server-measured probes? (Spearman, n=8)
probe server-rho: A 0.512, AB 0.638, L240 0.632, Z 0.608, S06 0.960, S07 0.784, S08 0.739, S09 0.481
  max|rho| excl >0.95   : 0.94, 0.83, 0.88, 0.24, 0.77, 0.91, 0.78, 0.93 | Spearman -0.48 (p 0.23)
  max|rho| excl >0.9    : 0.86, 0.83, 0.88, 0.24, 0.77, 0.48, 0.78, 0.61 | Spearman -0.10 (p 0.82)
  max|rho| excl >0.8    : 0.76, 0.79, 0.71, 0.24, 0.77, 0.48, 0.78, 0.61 | Spearman +0.33 (p 0.42)
  count |rho|>0.4       : 25, 65, 55, 0, 12, 9, 27, 13 | Spearman -0.07 (p 0.87)
  count |rho|>0.5       : 12, 42, 31, 0, 6, 4, 11, 4 | Spearman +0.07 (p 0.87)
  count |rho|>0.6       : 5, 25, 12, 0, 6, 4, 5, 2 | Spearman +0.38 (p 0.35)

## T5 - is the hourly correlation with stand-ins the same in both dispersion regimes?
share of high-dispersion hours 0.47; mean SWD weight on AB 0.47
AB: top-10 stand-ins, rho in high vs low dispersion:
   idio_F1_L120_K3@48       high +0.846  low +0.817  diff +0.029
   idio_F2_L120_K3@48       high +0.844  low +0.814  diff +0.029
   idio_F1_L120_K5@48       high +0.811  low +0.770  diff +0.041
   idio_F3_L120_K3@48       high +0.799  low +0.770  diff +0.029
   idio_F4_L120_K3@48       high +0.782  low +0.753  diff +0.028
   idio_F1_L120_K8@48       high +0.790  low +0.739  diff +0.051
   idio_F1_L48_K8@336       high +0.778  low +0.740  diff +0.038
   idio_F1_L48_K8@48        high +0.744  low +0.717  diff +0.028
   idio_F1_L120_K1@48       high +0.719  low +0.706  diff +0.014
   idio_F1_L48_K3@336       high +0.701  low +0.662  diff +0.038
Z: top-10 stand-ins, rho in high vs low dispersion:
   lvl3@336                 high +0.225  low +0.245  diff -0.020
   lvl2@336                 high +0.219  low +0.237  diff -0.018
   lvl1@336                 high +0.219  low +0.237  diff -0.018
   lvl4@336                 high +0.211  low +0.226  diff -0.015
   tsz6@336                 high -0.231  low -0.192  diff -0.039
   d120_3@336               high -0.201  low -0.193  diff -0.008
   d120_1@336               high -0.199  low -0.190  diff -0.009
   d120_2@336               high -0.198  low -0.191  diff -0.007
   az5@336                  high -0.187  low -0.194  diff +0.007
   d120_4@336               high -0.192  low -0.185  diff -0.007
rho(AB,Z): high +0.107 low +0.070

## Ratio transfer (ASSUMPTION: the true blocker relates to each candidate like the stand-ins near it do)
X vs AB-type blockers (53 stand-ins): ratio median 0.748 [p10 0.675, p90 0.813] -> predicted rho 0.477 [0.431, 0.519]
SWD vs AB-type blockers (53 stand-ins): ratio median 0.636 [p10 0.569, p90 0.735] -> predicted rho 0.406 [0.363, 0.469]
SWC vs AB-type blockers (53 stand-ins): ratio median 0.619 [p10 0.566, p90 0.671] -> predicted rho 0.395 [0.361, 0.428]

## Addendum (same day): slow-smoothed stand-ins (hl 672/1344) and dispersion-timing test
* Slow stand-ins nearest Z: slow level tilts (lvl1-4 @1344) rho +0.35, tsz6 @1344 -0.35; still none >= 0.40 ->
  Z's blocker (0.608) is a specific construction absent from the generic library; its cross-term with AB is NOT estimable locally.
* Dispersion timing (regime of hour t-1, net PnL, 31 periods): AB high-minus-low +0.032 per-hour SR units (t +2.44);
  Z +0.009 (t +0.88); AB-minus-Z regime advantage +0.024 (t +1.43, 61% periods +) -> timing plausible, NOT established.
