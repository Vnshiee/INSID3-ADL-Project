# Project State

_Last update: 2026-11-10. All experiments executed, evaluated, and verified on an NVIDIA GeForce RTX 3070 GPU (8 GB VRAM, CUDA 12.6, PyTorch 2.7.1). Codebase audited against raw outputs and prepared for review/release._

## Summary

| Item | State | Notes |
|---|:---:|---|
| **Official INSID3 baseline** | **Verified** | Vendored unmodified (commit `0c165a1`); 40/40 files checksum-verified |
| **Task 1 (Adaptive Debiasing)** | **Completed & Verified** | All 5 modes implemented and evaluated: Weighted, Spectral, Calibrated MMSE, Image-Dependent, Token-Dependent |
| **Task 2 (Multi-Prototype Matching)** | **Completed & Verified** | All aggregation and matching schemes implemented and evaluated: Max, Weighted, LSE, Chamfer, Unbalanced OT |
| **Unit Test Suite** | **126/126 Passed** | Full test suite (`pytest tests/ -q`) passing in 5.84s |
| **Empirical Calibration** | **Completed** | ViT-B (768d) and ViT-L (1024d) calibration on 500 COCO images |
| **Validation Tuning Sweeps** | **Completed** | 77 tuning runs (1 `baseline` + 29 `t1_*` + 42 `t2_*` + 5 `c_*`; 77,000 episodes) on COCO `train2014` |
| **Model Selection** | **Completed** | Automated via `select_best.py` (`sel_task1`, `sel_task2`, `sel_combined`) |
| **Test Stage Benchmarks** | **Completed** | 14 complete 4-fold test evaluations (56 fold runs; 87,213 episodes) across COCO-20ⁱ and PASCAL-Part (1-shot & 5-shot) |
| **Semantic Correspondence** | **Completed** | 7 SPair-71k evaluations across all 12,234 test pairs each (85,638 pairs total) |
| **Statistical Significance** | **Completed** | Fold-stratified paired bootstrap resampling (5,000 resamples) with 95% CIs and two-sided $p$-values |
| **Documentation & Analysis** | **Completed** | Master journey report (`WhatWeDid.md`), `docs/RESULTS.md`, and failure analysis CSVs |

## Key Findings Record

1. **Reproduction Check Passed:**
   * Published COCO-20ⁱ 1-shot baseline: ~57.6% (CRF full test).
   * Ours (Bilinear): **57.48% mIoU** on COCO `tune` (1,000 eps) and **56.13% mIoU** on COCO-20ⁱ full 4-fold test (4,000 eps).
   * No debiasing baseline: **53.13% mIoU** on tune ($\Delta = -4.34\%$, $p=0.000$) and **53.86% mIoU** on COCO-20ⁱ test ($\Delta = -2.27\%$, $p=0.000$), confirming positional debiasing is essential.
   * Equivalence sanity check (`t2_check_m1` vs `baseline`): exact identity ($\Delta = 0.00\%$, `57.4772` vs `57.4772`).
   * SPair-71k correspondence (ViT-B 768, $s=20$): original **30.08 / 46.85 / 55.62 / 61.26** (vs 30.1 / 46.8 / 55.6 / 61.2 published); hard debiased **33.68 / 52.57 / 62.51 / 68.73** (vs 33.7 / 52.6 / 62.5 / 68.7 published).

2. **Task 1: Spectral Wiener Attenuation Wins on Tuning & Matches on Test:**
   * `t1_spectral_k500_g4` ($k=500, \gamma=4$): **57.92% mIoU** ($\mathbf{+0.44\%}$, 95% CI $[+0.02\%, +1.02\%]$, $p = \mathbf{0.040}$).
   * Small GT-area tercile (`gt_area <= 26,530` px at $1024^2$, $n=334$): **$+0.98\%$ mean foreground IoU gain** (`48.09%` vs `47.11%`).
   * Full test benchmarks: **56.13%** on COCO-20ⁱ ($-0.00\%$, $p=0.977$) and **49.05%** on PASCAL-Part ($-0.07\%$, $p=0.399$).
   * On SPair-71k semantic correspondence (point PCK@0.10): `token_ls` reaches **52.86%** ($+0.29\%$) and `calibrated_b1` reaches **52.73%** ($+0.16\%$) vs `52.57%` hard baseline.

3. **Task 2: Multi-Prototype Dynamics & Merge Threshold Interaction:**
   * Log-Sum-Exp aggregation ($M=2, T=0.05$): **57.56% mIoU** on tune ($+0.08\%$, $p=0.951$; **$+0.60\%$** on medium GT-area tercile `60.57%` vs `59.97%`), **55.95%** on COCO-20ⁱ test ($-0.19\%$, $p=0.312$), and **48.51%** on PASCAL-Part test ($-0.62\%$, $p=0.000$).
   * Identified and verified the merge threshold confound: increasing $M$ inflates max similarity scores causing background bleed under default $\alpha=0.20$, while naively raising $\alpha \in \{0.25, 0.30, 0.35\}$ reduces recall (`54.02%`, `52.17%`, `49.20%` for `max` $M=4$).
   * Unbalanced Optimal Transport Sinkhorn matching ($\epsilon=0.02, \rho=0.1, M=4$): **57.32% mIoU** ($\Delta = -0.15\%$, $p=0.785$).

4. **Resource Profile (RTX 3070 8 GB VRAM):**
   * 1-shot latency & memory: baseline `546–548 ms` / `1,562 MB`; extensions `544–555 ms` / `1,548–1,564 MB`.
   * 5-shot latency & memory: baseline `1,190 ms` / `2,258 MB`; combined `1,202–1,205 ms` / `2,260 MB`.
