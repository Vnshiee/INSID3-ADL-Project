# Selection report (COCO train2014 tuning episodes; test sets untouched)

Baseline tune mIoU: 57.48

## Task 1

| rank | exp | mIoU |
|---|---|---|
| 1 | t1_spectral_k500_g4 | 57.92 |
| 2 | t1_spectral_k500_g1 | 57.84 |
| 3 | t1_weighted_a0.9 | 57.45 |
| 4 | t1_spectral_k300_g4 | 57.29 |
| 5 | t1_spectral_k300_g1 | 57.21 |
| 6 | t1_weighted_a0.75 | 57.15 |
| 7 | t1_hard_s300 | 57.15 |
| 8 | t1_hard_s700 | 57.12 |
| 9 | t1_token_ls_b1.0 | 57.06 |
| 10 | t1_token_fixed_b1.0 | 57.01 |
| 11 | t1_calibrated_b2.0 | 56.93 |
| 12 | t1_hard_s100 | 56.89 |
| 13 | t1_spectral_k700_g4 | 56.87 |
| 14 | t1_spectral_k700_g1 | 56.87 |
| 15 | t1_calibrated_b1.0 | 56.87 |
| 16 | t1_calibrated_b1.5 | 56.79 |
| 17 | t1_image_cal_b2.0 | 56.78 |
| 18 | t1_token_fixed_b0.5 | 56.73 |
| 19 | t1_weighted_a0.5 | 56.71 |
| 20 | t1_image_cal_b1.0 | 56.69 |
| 21 | t1_image_noise_b2.0 | 56.66 |
| 22 | t1_image_noise_b0.5 | 56.52 |
| 23 | t1_image_noise_b1.0 | 56.48 |
| 24 | t1_token_ls_b0.5 | 56.47 |
| 25 | t1_image_cal_b0.5 | 55.97 |
| 26 | t1_hard_s900 | 55.82 |
| 27 | t1_calibrated_b0.5 | 55.68 |
| 28 | t1_weighted_a0.25 | 54.80 |
| 29 | t1_none | 53.13 |

## Task 2

| rank | exp | mIoU |
|---|---|---|
| 1 | t2_lse_m2_t0.05 | 57.56 |
| 2 | t2_check_m1 | 57.48 |
| 3 | t2_ot_m4_e0.02_r0.1 | 57.32 |
| 4 | t2_max_m2 | 57.25 |
| 5 | t2_weighted_m2 | 57.20 |
| 6 | t2_lse_m3_t0.05 | 57.19 |
| 7 | t2_ot_m4_e0.05_r0.1 | 57.05 |
| 8 | t2_weighted_m3 | 57.03 |
| 9 | t2_ot_m4_e0.02_r0.5 | 57.00 |
| 10 | t2_chamfer_m4 | 56.95 |
| 11 | t2_lse_m4_t0.1 | 56.93 |
| 12 | t2_ot_m4_e0.05_r0.5 | 56.91 |
| 13 | t2_ot_m4_e0.1_r0.1 | 56.87 |
| 14 | t2_lse_m4_t0.02 | 56.84 |
| 15 | t2_ot_m4_e0.02_r100.0 | 56.79 |
| 16 | t2_ot_m4_e0.02_r2.0 | 56.76 |
| 17 | t2_agglo_tau0.5_max | 56.71 |
| 18 | t2_lse_m4_t0.05 | 56.69 |
| 19 | t2_weighted_m4 | 56.64 |
| 20 | t2_max_m4_global | 56.60 |
| 21 | t2_weighted_m6 | 56.50 |
| 22 | t2_ot_m4_e0.05_r100.0 | 56.48 |
| 23 | t2_lse_m6_t0.05 | 56.48 |
| 24 | t2_ot_m4_e0.05_r2.0 | 56.47 |
| 25 | t2_max_m3 | 56.39 |
| 26 | t2_lse_m8_t0.05 | 56.38 |
| 27 | t2_agglo_tau0.6_max | 56.31 |
| 28 | t2_weighted_m8 | 56.31 |
| 29 | t2_ot_m4_e0.1_r0.5 | 56.21 |
| 30 | t2_max_m4_groupdeb | 56.03 |
| 31 | t2_ot_m4_e0.1_r2.0 | 56.03 |
| 32 | t2_ot_m4_e0.1_r100.0 | 55.97 |
| 33 | t2_max_m4 | 55.94 |
| 34 | t2_agglo_tau0.7_max | 55.90 |
| 35 | t2_max_m8 | 55.82 |
| 36 | t2_max_m6 | 55.79 |
| 37 | t2_lse_m4_alpha0.25 | 54.55 |
| 38 | t2_max_m4_alpha0.25 | 54.02 |
| 39 | t2_lse_m4_alpha0.3 | 52.23 |
| 40 | t2_max_m4_alpha0.3 | 52.17 |
| 41 | t2_max_m4_alpha0.35 | 49.20 |
| 42 | t2_lse_m4_alpha0.35 | 48.02 |

## Combined

| rank | exp | mIoU |
|---|---|---|
| 1 | c_hard_debias | 57.56 |
| 2 | c_group_debiased | 57.53 |
| 3 | c_selected | 57.45 |
| 4 | c_alpha0.15 | 57.36 |
| 5 | c_alpha0.25 | 55.52 |

