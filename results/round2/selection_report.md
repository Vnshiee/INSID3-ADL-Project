# Selection report (COCO train2014 tuning episodes; test sets untouched)

Baseline tune mIoU: 56.81

Note: differences smaller than the paired-bootstrap CI of the tuning set are not evidence; check results/comparisons.md before treating rank 1 as better than the baseline.

## Task 1

| rank | exp | mIoU | data |
|---|---|---|---|
| 1 | t1_spectral_k500_g4 | 56.73 | seeds=1, episodes=4000 |
| 2 | t1_hard_s50 | 56.36 | seeds=1, episodes=4000 |
| 3 | t1_calibrated_b1.0 | 56.22 | seeds=1, episodes=4000 |
| 4 | t1_hard_evalbasis | 56.21 | seeds=1, episodes=4000 |
| 5 | t1_hard_s10 | 56.12 | seeds=1, episodes=4000 |
| 6 | t1_rho_t0.3 | 56.11 | seeds=1, episodes=4000 |
| 7 | t1_hard_s20 | 55.99 | seeds=1, episodes=4000 |
| 8 | t1_hard_s5 | 55.94 | seeds=1, episodes=4000 |
| 9 | t1_center_hard_s500 | 55.53 | seeds=1, episodes=4000 |
| 10 | t1_pca_s20 | 54.62 | seeds=1, episodes=4000 |
| 11 | t1_random_s500 | 53.35 | seeds=1, episodes=4000 |
| 12 | t1_none | 53.21 | seeds=1, episodes=4000 |
| 13 | t1_center_only | 52.14 | seeds=1, episodes=4000 |
| 14 | t1_pca_s500 | 34.23 | seeds=1, episodes=4000 |

## Task 2

| rank | exp | mIoU | data |
|---|---|---|---|
| 1 | t2_lse_m2 | 56.50 | seeds=1, episodes=4000 |
| 2 | t2_lse_m4_alpha0.15 | 56.46 | seeds=1, episodes=4000 |
| 3 | t2_lse_adaptive100 | 56.45 | seeds=1, episodes=4000 |
| 4 | t2_lse_m4_scal | 55.83 | seeds=1, episodes=4000 |
| 5 | t2_max_m4_alpha0.15 | 55.76 | seeds=1, episodes=4000 |
| 6 | t2_max_m4_scal | 55.24 | seeds=1, episodes=4000 |
| 7 | t2_lse_m4_alpha0.1 | 55.05 | seeds=1, episodes=4000 |
| 8 | t2_max_m4_alpha0.1 | 54.24 | seeds=1, episodes=4000 |

## Combined

| rank | exp | mIoU | data |
|---|---|---|---|

