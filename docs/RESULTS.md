# Results

> **STATUS: EXECUTED & FULLY COMPILED.**  
> Experiments were executed on an NVIDIA GeForce RTX 3070 GPU (8 GB VRAM, CUDA 12.6, PyTorch 2.7.1) using the frozen DINOv3 backbone. All metrics below are filled strictly from the outputs of `./evaluate.sh` ([`results/summary.md`](../results/summary.md), [`results/summary_segmentation_runs.csv`](../results/summary_segmentation_runs.csv), [`results/comparisons.md`](../results/comparisons.md), [`results/selection_report.md`](../results/selection_report.md)) and the SPair-71k `metrics.json` outputs under `outputs/corr_base768_*/spair/test_seed0/` on paired episodes with full RNG isolation.  
> *Note on rounding:* $\Delta$ values vs. `baseline` match [`results/comparisons.md`](../results/comparisons.md) (computed from unrounded run metrics prior to two-decimal rounding).

---

## R0 — Baseline Reproduction

| Setting | Published (CVPR 2026) | Ours (Bilinear) | Ours (CRF) | Notes / Sanity Check |
|---|:---:|:---:|:---:|---|
| **COCO-20ⁱ 1-shot mIoU (test, 4 folds)** | 57.6 (CRF, full test) | **56.13** (full 4-fold test) / **57.48** (tune split) | — | Test FB-IoU = 76.02% (4,000 eps); Tune FB-IoU = 76.74% (1,000 eps) |
| **COCO-20ⁱ No Debiasing ($s=0$)** | 54.5 (full test) | **53.86** (full 4-fold test) / **53.13** (tune split) | — | Test $\Delta = -2.27\%$ ($p=0.000$); Tune $\Delta = -4.34\%$ ($p=0.000$) |
| **PASCAL-Part 1-shot mIoU (test, 4 folds)** | 50.5 (CRF, full test) | **49.13** (full 4-fold test) | — | 8,459 test episodes across 4 folds; FB-IoU = 68.90% |
| **SPair-71k ViT-B 768 ($s=20$), Original PCK** | 30.1 / 46.8 / 55.6 / 61.2 | **30.08 / 46.85 / 55.62 / 61.26** | n/a | Evaluated across all 12,234 test pairs ($\pm 0.06\%$ match) |
| **SPair-71k ViT-B 768 ($s=20$), Debiased (Hard)** | 33.7 / 52.6 / 62.5 / 68.7 | **33.68 / 52.57 / 62.51 / 68.73** | n/a | Evaluated across all 12,234 test pairs ($\pm 0.03\%$ match) |
| **Sanity Equivalence (`t2_check_m1` − `baseline`)** | must be 0.00 | **0.00** (57.48 vs 57.48) | — | Exact numerical identity check passed (`57.4772` vs `57.4772`) |

---

## R1 — Task 1: Adaptive Positional Debiasing on Tuning Episodes (COCO `train2014`, 1,000 episodes)

| Mode | Parameters / Setting | mIoU (%) | $\Delta$ vs Hard | 95% CI | $p$-value |
|---|---|:---:|:---:|:---:|:---:|
| **none** | $s = 0$ (no debiasing) | 53.13 | $-4.34$ | $[-5.91, -2.81]$ | $0.000$ |
| **hard (official)** | $s = 500$ (orthogonal complement) | 57.48 | $0.00$ | — | — |
| **hard** | $s = 100$ | 56.89 | $-0.58$ | $[-1.77, +0.47]$ | $0.266$ |
| **hard** | $s = 300$ | 57.15 | $-0.33$ | $[-1.37, +0.66]$ | $0.474$ |
| **hard** | $s = 700$ | 57.12 | $-0.36$ | $[-1.40, +0.81]$ | $0.606$ |
| **hard** | $s = 900$ | 55.82 | $-1.66$ | $[-3.33, -0.09]$ | $0.040$ |
| **weighted** | $\alpha = 0.25$ | 54.80 | $-2.67$ | $[-3.98, -1.57]$ | $0.000$ |
| **weighted** | $\alpha = 0.50$ | 56.71 | $-0.77$ | $[-1.87, +0.03]$ | $0.056$ |
| **weighted** | $\alpha = 0.75$ | 57.15 | $-0.32$ | $[-1.04, +0.27]$ | $0.268$ |
| **weighted** | $\alpha = 0.90$ | 57.45 | $-0.03$ | $[-0.14, +0.14]$ | $0.794$ |
| **spectral** | $k=300, \gamma=1$ | 57.21 | $-0.27$ | $[-1.22, +0.63]$ | $0.542$ |
| **spectral** | $k=300, \gamma=4$ | 57.29 | $-0.19$ | $[-1.23, +0.76]$ | $0.640$ |
| **spectral** | $k=500, \gamma=1$ | 57.84 | $+0.37$ | $[-0.22, +1.06]$ | $0.204$ |
| **spectral (Best Task 1)** | $k=500, \gamma=4$ | **57.92** | $\mathbf{+0.44}$ | $[\mathbf{+0.02}, \mathbf{+1.02}]$ | $\mathbf{0.040}$ |
| **spectral** | $k=700, \gamma=1$ | 56.87 | $-0.61$ | $[-1.57, +0.62]$ | $0.394$ |
| **spectral** | $k=700, \gamma=4$ | 56.87 | $-0.60$ | $[-1.61, +0.55]$ | $0.330$ |
| **calibrated** | $\beta = 0.5$ | 55.68 | $-1.79$ | $[-3.07, -0.64]$ | $0.002$ |
| **calibrated** | $\beta = 1.0$ (MMSE default) | 56.87 | $-0.61$ | $[-1.77, +0.40]$ | $0.216$ |
| **calibrated** | $\beta = 1.5$ | 56.79 | $-0.68$ | $[-1.78, +0.34]$ | $0.199$ |
| **calibrated** | $\beta = 2.0$ | 56.93 | $-0.55$ | $[-1.66, +0.44]$ | $0.264$ |
| **image** | $\beta = 0.5$ (calibrated variance) | 55.97 | $-1.51$ | $[-2.80, -0.38]$ | $0.006$ |
| **image** | $\beta = 1.0$ (calibrated variance) | 56.69 | $-0.79$ | $[-1.96, +0.20]$ | $0.114$ |
| **image** | $\beta = 2.0$ (calibrated variance) | 56.78 | $-0.69$ | $[-1.79, +0.25]$ | $0.146$ |
| **image** | $\beta = 0.5$ (noise variance) | 56.52 | $-0.96$ | $[-2.16, +0.03]$ | $0.060$ |
| **image** | $\beta = 1.0$ (noise variance) | 56.48 | $-0.99$ | $[-2.15, +0.04]$ | $0.056$ |
| **image** | $\beta = 2.0$ (noise variance) | 56.66 | $-0.82$ | $[-1.97, +0.18]$ | $0.108$ |
| **token** | $\beta = 0.5$ (least-squares gain) | 56.47 | $-1.01$ | $[-2.29, +0.03]$ | $0.057$ |
| **token** | $\beta = 1.0$ (least-squares gain) | 57.06 | $-0.41$ | $[-1.53, +0.68]$ | $0.460$ |
| **token** | $\beta = 0.5$ (fixed gain) | 56.73 | $-0.75$ | $[-1.95, +0.26]$ | $0.135$ |
| **token** | $\beta = 1.0$ (fixed gain) | 57.01 | $-0.46$ | $[-1.87, +0.88]$ | $0.513$ |

---

## R2 — Task 1: Semantic Correspondence (SPair-71k Test, 12,234 Image Pairs)

Evaluated with DINOv3 ViT-B/16 at 768 px resolution ($s=20$). *(Note: `scripts/compare_all.py` runs bootstrap comparisons on segmentation only, so SPair-71k rows report point PCK deltas without paired bootstrap CIs).*

| Debiasing Mode | Setting | PCK@0.05 (%) | PCK@0.10 (%) | PCK@0.15 (%) | PCK@0.20 (%) | $\Delta$@0.10 vs Hard |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **original (no debias)** | `debiased=false` | 30.08 | 46.85 | 55.62 | 61.26 | $-5.72$ |
| **hard (official)** | $s = 20$ | 33.68 | 52.57 | 62.51 | 68.73 | $0.00$ (reference) |
| **weighted** | $\alpha = 0.5$ | 33.11 | 51.89 | 61.60 | 67.64 | $-0.68$ |
| **spectral** | $k=20, \gamma=1$ | 33.66 | 52.54 | 62.56 | 68.80 | $-0.03$ |
| **image** | $\beta = 1.0$ | 33.75 | 52.67 | 62.50 | 68.73 | $+0.10$ |
| **calibrated** | $\beta = 1.0$ | 33.83 | **52.73** | **62.61** | **68.81** | **+0.16** |
| **token (Best Corr.)** | least-squares | **34.21** | **52.86** | 62.38 | 68.36 | **+0.29** |

---

## R3 — Task 2: Multi-Prototype Reference Matching on Tuning Episodes (COCO `train2014`, 1,000 episodes)

Evaluation across prototype counts $M \in \{1, 2, 3, 4, 6, 8\}$ and aggregation operators:

| Aggregation \ $M$ | $M=1$ | $M=2$ | $M=3$ | $M=4$ | $M=6$ | $M=8$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **max** | 57.48 | 57.25 | 56.39 | 55.94 | 55.79 | 55.82 |
| **weighted** | 57.48 | 57.20 | 57.03 | 56.64 | 56.50 | 56.31 |
| **lse ($T=0.05$)** | 57.48 | **57.56** | 57.19 | 56.69 | 56.48 | 56.38 |

### Additional Task 2 Variants & Many-to-Many Matching:
* **LSE Temperature Sweep ($M=4$):**
  * $T = 0.02$: 56.84% mIoU
  * $T = 0.05$: 56.69% mIoU
  * $T = 0.10$: 56.93% mIoU
* **Agglomerative Clustering (Adaptive $M$):**
  * $\tau = 0.5$: 56.71% mIoU
  * $\tau = 0.6$: 56.31% mIoU
  * $\tau = 0.7$: 55.90% mIoU
* **Many-to-Many Matching ($M=4$):**
  * **Symmetric Chamfer:** 56.95% mIoU ($\Delta = -0.53\%$)
  * **Unbalanced Optimal Transport ($\epsilon=0.02, \rho=0.1$):** **57.32% mIoU** ($\Delta = -0.15\%$, 95% CI $[-1.09, +0.84]$, $p = 0.785$)
  * **Unbalanced Optimal Transport ($\epsilon=0.02, \rho=0.5$):** 57.00% mIoU
  * **Unbalanced Optimal Transport ($\epsilon=0.05, \rho=0.1$):** 57.05% mIoU
* **Merge Threshold Interaction ($\alpha$ sweep for $M=4$):**
  * $\alpha = 0.20$ (default): 55.94% (`max`) / 56.69% (`lse`)
  * $\alpha = 0.25$: 54.02% (`max`) / 54.55% (`lse`)
  * $\alpha = 0.30$: 52.17% (`max`) / 52.23% (`lse`)
  * $\alpha = 0.35$: 49.20% (`max`) / 48.02% (`lse`)  
  *(Reveals that shifting $\alpha$ higher without re-calibrating reduces recall).*

---

## R4 — Selected Configurations Summary (COCO `train2014` Tuning Split)

Selected best configurations determined strictly on `coco-tune` (saved to [`configs/selected/`](../configs/selected/)):

| Model / Configuration | Method Details | Tuning mIoU (%) | $\Delta$ vs Baseline |
|---|---|:---:|:---:|
| **Baseline** | Hard orthogonal complement projection ($s=500$), single prototype | 57.48 | $0.00$ |
| **Task 1 Selected** | Spectral Wiener attenuation (`t1_spectral_k500_g4`, $k=500, \gamma=4$) | **57.92** | $\mathbf{+0.44}$ |
| **Task 2 Selected** | Multi-prototype Log-Sum-Exp (`t2_lse_m2_t0.05`, $M=2, T=0.05$) | **57.56** | $\mathbf{+0.08}$ |
| **Combined (`c_selected`)** | `sel_task1` + `sel_task2` (Spectral + LSE $M=2$) | 57.45 | $-0.03$ |
| **Combined (`c_hard_debias`)** | Hard debiasing + Multi-prototype LSE ($M=2$) | **57.56** | $+0.08$ |
| **Combined (`c_group_debiased`)** | `sel_task1` + `sel_task2` grouping in debiased space | 57.53 | $+0.06$ |

---

## R5 — Full 4-Fold Test Benchmarks (COCO-20ⁱ & PASCAL-Part)

Evaluated across all 4 official test folds on COCO-20ⁱ (1,000 episodes/fold = 4,000 total) and PASCAL-Part (2,500 / 959 / 2,500 / 2,500 = 8,459 total). Note: `sel_combined` uses `configs/selected/combined.yaml` (`spectral k=500, \gamma=4` + `lse M=2, T=0.05`).

| Dataset | Shots | Experiment | Folds | Mean mIoU (%) | Per-Fold [F0, F1, F2, F3] | Mean FB-IoU (%) | $\Delta$ vs Baseline | 95% CI | $p$-value |
|---|:---:|---|:---:|:---:|---|:---:|:---:|:---:|:---:|
| **COCO-20ⁱ** | 1 | **`baseline`** | 4/4 | **56.13** | 54.9, 60.4, 54.4, 54.8 | 76.02 | — | — | — |
| **COCO-20ⁱ** | 1 | **`baseline_no_debias`** | 4/4 | 53.86 | 51.7, 57.3, 53.5, 52.8 | 74.80 | $-2.27$ | $[-3.06, -1.45]$ | $0.000$ |
| **COCO-20ⁱ** | 1 | **`sel_task1`** | 4/4 | **56.13** | 54.5, 60.4, 54.9, 54.7 | 76.05 | $-0.00$ | $[-0.27, +0.27]$ | $0.977$ |
| **COCO-20ⁱ** | 1 | **`sel_task2`** | 4/4 | 55.95 | 54.8, 59.6, 54.6, 54.8 | 75.90 | $-0.19$ | $[-0.56, +0.18]$ | $0.312$ |
| **COCO-20ⁱ** | 1 | **`sel_combined`** | 4/4 | 55.99 | 54.6, 60.0, 54.8, 54.6 | 75.90 | $-0.15$ | $[-0.56, +0.26]$ | $0.475$ |
| **COCO-20ⁱ** | 5 | **`baseline`** | 4/4 | **64.70** | 64.3, 66.9, 64.8, 62.9 | 80.63 | — | — | — |
| **COCO-20ⁱ** | 5 | **`sel_combined`** | 4/4 | 64.23 | 63.2, 66.8, 64.3, 62.6 | 80.16 | $-0.48$ | $[-0.87, -0.07]$ | $0.022$ |
| **PASCAL-Part** | 1 | **`baseline`** | 4/4 | **49.13** | 45.4, 53.2, 44.9, 52.9 | 68.90 | — | — | — |
| **PASCAL-Part** | 1 | **`baseline_no_debias`** | 4/4 | 48.07 | 43.8, 54.7, 42.8, 51.0 | 68.20 | $-1.05$ | $[-1.52, -0.57]$ | $0.000$ |
| **PASCAL-Part** | 1 | **`sel_task1`** | 4/4 | 49.05 | 45.4, 52.8, 44.9, 53.0 | 68.86 | $-0.07$ | $[-0.25, +0.09]$ | $0.399$ |
| **PASCAL-Part** | 1 | **`sel_task2`** | 4/4 | 48.51 | 45.2, 51.7, 44.3, 52.8 | 68.35 | $-0.62$ | $[-0.90, -0.34]$ | $0.000$ |
| **PASCAL-Part** | 1 | **`sel_combined`** | 4/4 | 48.55 | 45.3, 51.8, 44.3, 52.8 | 68.39 | $-0.58$ | $[-0.87, -0.29]$ | $0.000$ |
| **PASCAL-Part** | 5 | **`baseline`** | 4/4 | **57.58** | 51.1, 67.0, 51.6, 60.6 | 75.08 | — | — | — |
| **PASCAL-Part** | 5 | **`sel_combined`** | 4/4 | 55.73 | 51.0, 61.8, 50.2, 59.9 | 73.63 | $-1.85$ | $[-2.19, -1.52]$ | $0.000$ |

---

## R6 — Paired Statistical Significance vs. Baseline (Key Comparisons)

Computed via fold-stratified paired bootstrap resampling (5,000 resamples, from [`results/comparisons.md`](../results/comparisons.md)):

| Method | Dataset (Split) | Ref mIoU | Model mIoU | $\Delta$ mIoU | 95% Confidence Interval | $p$-value | Significance |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`t1_none`** | COCO (`tune`) | 57.48 | 53.13 | $-4.34$ | $[-5.91, -2.81]$ | $0.000$ | Statistically worse |
| **`t1_weighted_a0.25`** | COCO (`tune`) | 57.48 | 54.80 | $-2.67$ | $[-3.98, -1.57]$ | $0.000$ | Statistically worse |
| **`t1_calibrated_b0.5`** | COCO (`tune`) | 57.48 | 55.68 | $-1.79$ | $[-3.07, -0.64]$ | $0.002$ | Statistically worse |
| **`t1_spectral_k500_g1`** | COCO (`tune`) | 57.48 | 57.84 | $+0.37$ | $[-0.22, +1.06]$ | $0.204$ | Positive trend |
| **`t1_spectral_k500_g4`** | COCO (`tune`) | 57.48 | **57.92** | $\mathbf{+0.44}$ | $[\mathbf{+0.02}, \mathbf{+1.02}]$ | $\mathbf{0.040}$ | **Statistically significant ($p < 0.05$)** |
| **`t2_lse_m2_t0.05`** | COCO (`tune`) | 57.48 | **57.56** | $+0.08$ | $[-0.64, +0.70]$ | $0.951$ | Statistically equivalent |
| **`t2_ot_m4_e0.02_r0.1`** | COCO (`tune`) | 57.48 | 57.32 | $-0.15$ | $[-1.09, +0.84]$ | $0.785$ | Statistically equivalent |
| **`t2_max_m4`** | COCO (`tune`) | 57.48 | 55.94 | $-1.54$ | $[-2.78, -0.34]$ | $0.010$ | Statistically worse |
| **`c_selected`** | COCO (`tune`) | 57.48 | 57.45 | $-0.03$ | $[-0.84, +0.84]$ | $0.956$ | Statistically equivalent |
| **`c_hard_debias`** | COCO (`tune`) | 57.48 | **57.56** | $+0.08$ | $[-0.64, +0.70]$ | $0.951$ | Statistically equivalent |
| **`sel_task1`** | COCO (`test`, 4 folds) | 56.13 | **56.13** | $-0.00$ | $[-0.27, +0.27]$ | $0.977$ | Statistically tied |
| **`sel_task1`** | PASCAL-Part (`test`, 4 folds) | 49.13 | 49.05 | $-0.07$ | $[-0.25, +0.09]$ | $0.399$ | Statistically tied |
| **`sel_task2`** | COCO (`test`, 4 folds) | 56.13 | 55.95 | $-0.19$ | $[-0.56, +0.18]$ | $0.312$ | Statistically tied |
| **`sel_task2`** | PASCAL-Part (`test`, 4 folds) | 49.13 | 48.51 | $-0.62$ | $[-0.90, -0.34]$ | $0.000$ | Statistically worse |
| **`sel_combined`** | COCO (`test`, 5-shot) | 64.70 | 64.23 | $-0.48$ | $[-0.87, -0.07]$ | $0.022$ | Statistically worse |
| **`sel_combined`** | PASCAL-Part (`test`, 5-shot) | 57.58 | 55.73 | $-1.85$ | $[-2.19, -1.52]$ | $0.000$ | Statistically worse |

---

## R7 — Runtime and Memory Profile

Measured on NVIDIA GeForce RTX 3070 (8 GB VRAM) under CUDA 12.6 with DINOv3 ViT-L/16 backbone at 1024 px (`results/summary.md`):

| Method / Architecture | Algorithm Details | Latency (ms / episode) | Peak GPU Memory (MB) |
|---|---|:---:|:---:|
| **Baseline (Official, 1-shot)** | Hard projection ($s=500$), 1 prototype (COCO tune / test) | 546–548 ms | 1,562 MB |
| **Task 1: Spectral (1-shot)** | Closed-form Wiener schedule ($k=500, \gamma=4$) | 548 ms | 1,564 MB |
| **Task 1: Calibrated (1-shot)** | Empirical MMSE gain vector ($\beta=1.0$) | 547 ms | 1,564 MB |
| **Task 1: Token-wise (1-shot)** | Spatial template subtraction (`least_squares`, $\beta=1.0$) | 549 ms | 1,564 MB |
| **Task 2: Multi-Prototype LSE (1-shot)** | Spherical k-means ($M=2$) + LSE ($T=0.05$) | 546 ms | 1,548 MB |
| **Task 2: Multi-Prototype Max (1-shot)** | Spherical k-means ($M=4$) + Max aggregation | 547 ms | 1,548 MB |
| **Task 2: Unbalanced OT (1-shot)** | Symmetrical entropic Sinkhorn ($M=4$, $\epsilon=0.02, \rho=0.1$) | 555 ms | 1,548 MB |
| **Task 2: Agglomerative (1-shot)** | Hierarchical cosine distance clustering ($\tau=0.5$) | 551 ms | 1,548 MB |
| **Baseline (Official, 5-shot)** | Hard projection ($s=500$), 1 prototype (COCO / PASCAL-Part) | 1,190 ms | 2,258 MB |
| **Combined Selected (5-shot)** | `sel_task1` + `sel_task2` (COCO / PASCAL-Part) | 1,202–1,205 ms | 2,260 MB |

*All 1-shot variants introduce less than 10 ms of overhead per episode and operate comfortably under 1.6 GB of VRAM on an 8 GB GPU (~2.26 GB for 5-shot).*
