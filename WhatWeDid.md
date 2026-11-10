# End-to-End Project Walkthrough & Experimental Analysis

**Base Paper:** *INSID3: Training-Free In-Context Segmentation with DINOv3* (CVPR 2026)  
**Paper Authors:** Claudia Cuttano, Gabriele Trivigno, Christoph Reich, Daniel Cremers, Carlo Masone, Stefan Roth  
**Research Extensions:**
1. **Task 1:** Adaptive Positional Debiasing
2. **Task 2:** Multi-Prototype Reference Matching  

**Hardware Used:** NVIDIA GeForce RTX 3070 (8 GB VRAM), Ubuntu Linux, CUDA 12.6, PyTorch 2.7.1  

---

## Table of Contents
1. [Prologue: What Problem Am I Solving?](#1-prologue-what-problem-am-i-solving)
2. [Chapter 1: Baseline Codebase Verification & Pretrained Checkpoints](#2-chapter-1-baseline-codebase-verification--pretrained-checkpoints)
3. [Chapter 2: Setting Up the Workstation & Experiment Pipeline](#3-chapter-2-setting-up-the-workstation--experiment-pipeline)
4. [Chapter 3: Preparing the Benchmarks (SPair-71k, COCO 2014 & PASCAL-Part)](#4-chapter-3-preparing-the-benchmarks-spair-71k-coco-2014--pascal-part)
5. [Chapter 4: Baseline Reproduction — Verifying the Original Method](#5-chapter-4-baseline-reproduction--verifying-the-original-method)
6. [Chapter 5: Task 1 — Adaptive Positional Debiasing](#6-chapter-5-task-1--adaptive-positional-debiasing)
7. [Chapter 6: Task 2 — Multi-Prototype Reference Matching](#7-chapter-6-task-2--multi-prototype-reference-matching)
8. [Chapter 7: Combined Experiments — Putting Task 1 and Task 2 Together](#8-chapter-7-combined-experiments--putting-task-1-and-task-2-together)
9. [Chapter 8: Full Benchmark Test Evaluation (COCO-20ⁱ & PASCAL-Part Across All Folds)](#9-chapter-8-full-benchmark-test-evaluation-coco-20ⁱ--pascal-part-across-all-folds)
10. [Chapter 9: Honest Result Analysis — What Does It All Mean?](#10-chapter-9-honest-result-analysis--what-does-it-all-mean)
11. [Chapter 10: Summary Tables & Numerical Proof](#11-chapter-10-summary-tables--numerical-proof)

---

## 1. Prologue: What Problem Am I Solving?

Imagine showing a model a reference image of a cat with an outline around it, then giving it a brand-new target image of a different cat in another room and asking it to segment the cat in the new scene. This problem is **In-Context Segmentation** (or Few-Shot Segmentation).

### How people used to do it
Traditionally, few-shot segmentation required training specialized decoders, prompt encoders, or task-specific heads. Every new visual domain (such as part segmentation, medical imaging, or aerial scenes) risked overfitting without retraining.

### The INSID3 Approach
In **INSID3** (CVPR 2026), Cuttano et al. asked whether in-context segmentation can be done completely **training-free** (zero training, zero fine-tuning, no extra decoder) using only a frozen self-supervised foundation model (Meta's DINOv3). They showed that it works — once you deal with a major obstacle inside Vision Transformers: **Positional Bias**.

### What is "Positional Bias"?
Vision Transformers divide an image into a grid of square patch tokens. Across transformer layers, tokens accumulate strong spatial coordinate encodings. As a result, a patch at the top-left corner of one image resembles another patch at the top-left corner of a completely different image, regardless of semantic content. Left uncorrected, a reference cat in the center biases the model to highlight the center of the target image.

### How INSID3 addressed it — and why I extended it
To suppress positional bias, INSID3:
1. Passes a uniform blank image through DINOv3 (since there is no visual texture, the activations expose positional structure).
2. Runs Singular Value Decomposition (SVD) on those features and retains the top $s = 500$ singular directions.
3. Applies a **hard orthogonal complement projection**: completely zeroing out all 500 directions for every token in every real image.

While effective, this leaves two natural research questions (`topic.md`):
* **Task 1 (Adaptive Positional Debiasing):**  
  Zeroing out 500 directions indiscriminately acts like a blunt instrument. If subtle textures or boundary cues share components with those axes, hard zeroing discards valid semantics too. Can a soft, adaptive debiasing operator improve correspondence and segmentation?
* **Task 2 (Multi-Prototype Reference Matching):**  
  INSID3 compresses the entire annotated reference region into a **single average prototype vector**. Complex objects contain distinct visual parts (ears, fur torso, face, wheels, wings). Does clustering reference tokens into multiple complementary prototypes improve matching?

---

## 2. Chapter 1: Baseline Codebase Verification & Pretrained Checkpoints

To ensure strict experimental fairness, I kept the official INSID3 repository (`visinf/INSID3`, commit `0c165a1`) completely unmodified under `models/`, `utils/`, `datasets/`, `opts.py`, and `inference_*.py`, implementing all research extensions in an isolated package (`insid3_ext/`) alongside automated runners (`scripts/`) and tests (`tests/`).

### Pre-Flight Verification Checks:
1. **Upstream Byte-Identity Verification:** Running `python scripts/verify_upstream.py` confirmed all 40 upstream files match `visinf/INSID3@0c165a1` byte-for-byte.
2. **Pretrained Weight Integrity:** Verified SHA256 hashes of all three official DINOv3 checkpoints (`vits16`: `08c60483`, `vitb16`: `73cec8be`, `vitl16`: `8aa4cbdd`).
3. **Unit & Equivalence Tests:** Running `pytest tests/ -q` executed all **126 unit tests**, passing 100% (including strict mathematical equivalence checks ensuring that disabling extensions recovers the exact official forward pass).

---

## 3. Chapter 2: Setting Up the Workstation & Experiment Pipeline

All GPU runs were executed on a local workstation equipped with an **NVIDIA GeForce RTX 3070 (8 GB VRAM)**, Python 3.10, and PyTorch 2.7.1 (CUDA 12.6).

Before launching full sweeps, I completed the environment plumbing:
1. Configured `env.sh` with dataset, checkpoint, and output paths (`INSID3_DATA_ROOT`, `INSID3_OUTPUT_ROOT`, `INSID3_PRETRAIN_DIR`, `INSID3_DINOV3_REPO`) and verified GPU visibility via `scripts/check_env.py`.
2. Verified `tmux`, `pytest`, `curl`, and `gdown` availability so long multi-hour sweeps could run unattended and resume cleanly if interrupted.
3. Implemented SHA256 episode fingerprinting in [`insid3_ext/experiment.py`](insid3_ext/experiment.py) so every compared method evaluates the **exact same episodes in the exact same order** with isolated local RNGs.

---

## 4. Chapter 3: Preparing the Benchmarks (SPair-71k, COCO 2014 & PASCAL-Part)

To validate both semantic correspondence and segmentation (whole-object and part-level), I prepared three standard datasets:

### 1. SPair-71k (Semantic Keypoint Correspondence)
* **Content:** 12,234 test image pairs across 18 object categories with annotated semantic keypoints.
* **Footprint:** ~216 MB compressed.

### 2. COCO 2014 / COCO-20ⁱ (In-Context Object Segmentation)
* **Content:** `train2014` (~83k images, 13 GB) and `val2014` (~41k images, 6.2 GB) plus instance masks and official COCO-20ⁱ fold splits.

### 3. PASCAL-Part (Part-Level Segmentation)
* **Content:** PASCAL VOC 2010 extended with fine-grained part masks (e.g., wheels, eyes, wings, legs), ~1.5 GB compressed.

### Strict Separation Between Tuning and Test Sets
Hyperparameters must never be tuned on test episodes. Using `scripts/run_segmentation.py` with `data.split=tune`, I sampled a dedicated 1,000-episode tuning set exclusively from `COCO2014/train2014`, logging zero overlap with COCO-20ⁱ test images (`[tune] images shared with COCO-20i test folds: 0 (expected 0)`). All model selection in `configs/selected/` happened strictly on this tuning split before running any test-fold benchmark.

---

## 5. Chapter 4: Baseline Reproduction — Verifying the Original Method

Before trusting any extension result, I first tested whether my environment reproduced the official INSID3 baseline cleanly.

### Experiment
Loaded frozen DINOv3 ViT-L/16 at 1024×1024 resolution with the 500-dimensional blank-image SVD basis and bilinear mask upsampling, and evaluated all 1,000 COCO tuning episodes.

### Results
* **Ours (Bilinear, COCO tune):** **57.48% mIoU** (FB-IoU: **76.74%**).
* **Published Reference (Paper Table 1):** 57.6% mIoU (using DenseCRF refinement on full 4-fold test).
* **Sanity Check 1 — Removing Debiasing (`t1_none`, $s=0$):** Dropped to **53.13% mIoU** (FB-IoU: **74.23%**), a **$-4.34\%$** collapse ($p = 0.000$, 95% CI $[-5.91, -2.81]$). This confirms DINOv3 positional bias severely degrades raw feature matching.
* **Sanity Check 2 — Extension Code Equivalence (`t2_check_m1`, $M=1$):** Produced **57.48% mIoU** (`57.4772` vs `57.4772`, $\Delta = \mathbf{0.00\%}$), confirming the extended pipeline adds zero numerical drift when configured at $M=1$.

---

## 6. Chapter 5: Task 1 — Adaptive Positional Debiasing

### Design & Implementation
Rather than subtracting 100% of all 500 positional axes uniformly ($\lambda_k = 1$), I implemented a unified parametric operator:
$$\tilde{\mathbf{x}} = \text{normalize}\big(\mathbf{x} - \mathbf{U}_r (\boldsymbol{\lambda} \odot \mathbf{c})\big), \qquad \lambda_k \in [0, 1]$$
and evaluated five strategies across **29 Task 1 tuning runs** (plus the `baseline`):
1. **Weighted Projection:** Constant attenuation $\lambda_k = \alpha$ for $\alpha \in \{0.25, 0.50, 0.75, 0.90\}$.
2. **Spectral Attenuation (Data-free Wiener schedule):**
   $$\lambda_k = \frac{\sigma_k^{2\gamma}}{\sigma_k^{2\gamma} + \sigma_\text{knee}^{2\gamma}}$$
   sweeping $k \in \{300, 500, 700\}$ and sharpness $\gamma \in \{1, 4\}$.
3. **Calibrated MMSE Attenuation:** Estimated empirical positional energy fractions $\hat{\rho}_k \in [0, 1]$ on 500 COCO `train2014` images (no labels) and set $\lambda_k = \text{clamp}(\beta \hat{\rho}_k, 0, 1)$ for $\beta \in \{0.5, 1.0, 1.5, 2.0\}$.
4. **Image-Dependent Scaling:** Dynamically adjusted suppression per image based on estimated image positional variance.
5. **Token-Dependent Subtraction:** Subtracted an empirical spatial template token-by-token (using fixed or least-squares per-token gains).

---

### Task 1 Tuning Results (COCO `train2014`, 1,000 Paired Episodes)

| Rank | Method | Parameters | mIoU (%) | $\Delta$ vs Baseline | 95% Confidence Interval | $p$-value |
|:---:|---|---|:---:|:---:|:---:|:---:|
| 1 | **`t1_spectral_k500_g4`** | $k=500, \gamma=4$ | **57.92%** | $\mathbf{+0.44\%}$ | $[\mathbf{+0.02}, \mathbf{+1.02}]$ | $\mathbf{0.040}$ |
| 2 | `t1_spectral_k500_g1` | $k=500, \gamma=1$ | 57.84% | $+0.37\%$ | $[-0.22, +1.06]$ | 0.204 |
| — | **Baseline (Hard)** | $s=500$ | **57.48%** | $0.00\%$ | — | — |
| 3 | `t1_weighted_a0.9` | $\alpha = 0.90$ | 57.45% | $-0.03\%$ | $[-0.14, +0.14]$ | 0.794 |
| 4 | `t1_spectral_k300_g4` | $k=300, \gamma=4$ | 57.29% | $-0.19\%$ | $[-1.23, +0.76]$ | 0.640 |
| 5 | `t1_spectral_k300_g1` | $k=300, \gamma=1$ | 57.21% | $-0.27\%$ | $[-1.22, +0.63]$ | 0.542 |
| 6 | `t1_weighted_a0.75` | $\alpha = 0.75$ | 57.15% | $-0.32\%$ | $[-1.04, +0.27]$ | 0.268 |
| 7 | `t1_hard_s300` | $s = 300$ | 57.15% | $-0.33\%$ | $[-1.37, +0.66]$ | 0.474 |
| 8 | `t1_token_ls_b1.0` | $\beta = 1.0$, LS gain | 57.06% | $-0.41\%$ | $[-1.53, +0.68]$ | 0.460 |
| 9 | `t1_calibrated_b2.0` | $\beta = 2.0$ | 56.93% | $-0.55\%$ | $[-1.66, +0.44]$ | 0.264 |
| 10 | `t1_calibrated_b1.0` | $\beta = 1.0$ (MMSE) | 56.87% | $-0.61\%$ | $[-1.77, +0.40]$ | 0.216 |
| 11 | `t1_image_cal_b2.0` | $\beta = 2.0$ | 56.78% | $-0.69\%$ | $[-1.79, +0.25]$ | 0.146 |
| 12 | `t1_weighted_a0.5` | $\alpha = 0.50$ | 56.71% | $-0.77\%$ | $[-1.87, +0.03]$ | 0.056 |
| 13 | `t1_weighted_a0.25` | $\alpha = 0.25$ | 54.80% | $-2.67\%$ | $[-3.98, -1.57]$ | 0.000 |
| 14 | `t1_none` | $s = 0$ | 53.13% | $-4.34\%$ | $[-5.91, -2.81]$ | 0.000 |

*(Representative subset from all 29 Task 1 tuning runs; the complete ranked table is in `results/selection_report.md`, and $\Delta$ values match `results/comparisons.md` from unrounded metrics.)*

---

### Deep Analysis of Task 1

1. **Spectral Attenuation Won on Tuning:**  
   `t1_spectral_k500_g4` reached **57.92% mIoU**, improving over hard projection by **$+0.44\%$**.
2. **Statistical Significance:**  
   Under fold-stratified paired bootstrap testing (5,000 resamples across 1,000 identical episodes), $p = \mathbf{0.040}$ ($p < 0.05$) with a positive 95% CI ($[+0.02, +1.02]$).
3. **Where the Gain Came From (GT-Area Tercile Breakdown):**  
   Using `scripts/analyze_failures.py` on `results/fail_spectral_k500_g4.csv`:
   - **Small Objects (bottom tercile, `gt_area <= 26,530` px at $1024^2$, $n=334$):** Mean $\Delta\text{IoU} = \mathbf{+0.98\%}$ (**48.09% vs 47.11%**).
   - **Medium Objects (middle tercile, $n=333$):** Mean $\Delta\text{IoU} = +0.36\%$ (60.32% vs 59.97%).
   - **Large Objects (top tercile, $n=333$):** Mean $\Delta\text{IoU} = +0.11\%$ (67.02% vs 66.91%).
   - **Episodes predicting a larger mask (2.8% of episodes, $n=28$):** Mean $\Delta\text{IoU} = \mathbf{+5.43\%}$.
   
   *Why:* Small objects depend heavily on fine high-frequency boundary signals. Softening the hard rank-500 cutoff preserves details that hard zeroing strips away.
4. **Why Calibrated MMSE ($56.87\%$) Did Not Beat Spectral ($57.92\%$):**  
   Calibration (`calib_large_1024_summary.json`) measured an empirical mean positional share $\hat{\rho} \approx 0.083$ across the top 500 axes (`57.4%` of total natural-image feature energy resides along those axes). Because MMSE minimizes linear reconstruction error in feature space rather than downstream thresholded IoU, it under-suppresses positional bias relative to hard/spectral schedules.
5. **Semantic Keypoint Matching on SPair-71k (12,234 Pairs):**  
   Point PCK@0.10 evaluations across all 12,234 test pairs (bootstrap CIs were not run on correspondence in `compare_all.py`):
   - Original features (no debiasing): **46.85%** PCK@0.10.
   - Hard projection baseline: **52.57%** PCK@0.10.
   - **Calibrated MMSE (`calibrated_b1`):** **52.73%** ($+0.16\%$).
   - **Token-wise Debiasing (`token_ls`):** **52.86%** ($+0.29\%$).

---

## 7. Chapter 6: Task 2 — Multi-Prototype Reference Matching

### Design & Implementation
Instead of pooling all foreground reference tokens into one vector, I implemented:
1. **Spherical $k$-means** (and agglomerative clustering) on the unit sphere to extract $M \in \{2, 3, 4, 6, 8\}$ prototypes $\{q_1, \dots, q_M\}$.
2. **Aggregation Operators:**
   - Maximum similarity: $a_i = \max_j \langle x_i, q_j \rangle$
   - Area-weighted similarity: $a_i = \sum_j w_j \langle x_i, q_j \rangle$
   - Log-Sum-Exp (smooth max): $a_i = T \log \sum_j w_j \exp(\langle x_i, q_j \rangle / T)$
3. **Many-to-Many Region Matching:**
   - Symmetric Chamfer similarity between target candidate clusters and reference prototypes.
   - Unbalanced entropic Optimal Transport via Sinkhorn iterations with marginal relaxation $\rho$.

---

### Task 2 Tuning Results (COCO `train2014`, 1,000 Episodes)

| Aggregation Method | $M=1$ (Baseline) | $M=2$ | $M=3$ | $M=4$ | $M=6$ | $M=8$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **max** | 57.48% | 57.25% | 56.39% | 55.94% | 55.79% | 55.82% |
| **weighted** | 57.48% | 57.20% | 57.03% | 56.64% | 56.50% | 56.31% |
| **lse ($T=0.05$)** | 57.48% | **57.56%** | 57.19% | 56.69% | 56.48% | 56.38% |

#### Many-to-Many Matching & Variants ($M=4$):
* **Symmetric Chamfer Matching:** 56.95% mIoU ($\Delta = -0.53\%$)
* **Unbalanced Optimal Transport ($\epsilon=0.02, \rho=0.1$):** **57.32% mIoU** ($\Delta = -0.15\%$, $p = 0.785$)
* **Unbalanced Optimal Transport ($\epsilon=0.02, \rho=0.5$):** 57.00% mIoU
* **Unbalanced Optimal Transport ($\epsilon=0.05, \rho=0.1$):** 57.05% mIoU
* **Agglomerative Clustering ($\tau=0.5$, adaptive $M$):** 56.71% mIoU

---

### Deep Analysis of Task 2: Why High $M$ Underperformed

* **$M=2$ with Log-Sum-Exp achieved 57.56%** ($\mathbf{+0.08\%}$, $p=0.951$ — statistically equivalent overall), improving the medium GT-area tercile ($n=333$) by **$+0.60\%$** (`60.57% vs 59.97%`).
* Above $M=2$, accuracy steadily dropped (~55.8% for `max`), driven by two mechanisms:

1. **Object-vs-Part Granularity Mismatch:** Splitting a whole COCO reference object into 4–8 isolated sub-part prototypes lets any background patch resembling a single sub-part (e.g., a roof looking like a plane wing) score high under `max`.
2. **The Merge Threshold Confound:** Because $\max_{j=1}^M \langle x_i, q_j \rangle \ge \langle x_i, \bar{q} \rangle$, multi-prototype matching shifts similarity scores upward against INSID3's fixed threshold $\alpha = 0.20$. Yet simply raising $\alpha$ at $M=4$ cuts recall sharply:
   * $\alpha = 0.20$: **55.94%** (`max`) / **56.69%** (`lse`)
   * $\alpha = 0.25$: **54.02%** (`max`) / **54.55%** (`lse`)
   * $\alpha = 0.30$: **52.17%** (`max`) / **52.23%** (`lse`)
   * $\alpha = 0.35$: **49.20%** (`max`) / **48.02%** (`lse`)

---

## 8. Chapter 7: Combined Experiments — Putting Task 1 and Task 2 Together

Evaluating combinations of the best Task 1 and Task 2 settings on the 1,000 tuning episodes:

| Experiment Name | Configuration | mIoU (%) | $\Delta$ vs Baseline |
|---|---|:---:|:---:|
| **Baseline** | Hard debiasing + Single prototype | 57.48% | $0.00\%$ |
| **`c_hard_debias`** | Hard debiasing + Multi-prototype LSE ($M=2$) | **57.56%** | $\mathbf{+0.08\%}$ |
| **`c_group_debiased`** | Grouping tokens in debiased space ($M=2$) | 57.53% | $+0.06\%$ |
| **`c_selected`** | Spectral debiasing + Multi-prototype LSE ($M=2$) | 57.45% | $-0.03\%$ |

Combining Spectral debiasing (`t1_spectral_k500_g4`) with LSE $M=2$ (`c_selected`) yielded 57.45% ($-0.03\%$, $p=0.956$) — slightly below each component individually, indicating non-additivity between soft debiasing variance and unsupervised k-means clustering.

---

## 9. Chapter 8: Full Benchmark Test Evaluation (COCO-20ⁱ & PASCAL-Part Across All Folds)

After locking selections on the tuning split, I ran the full 4-fold test evaluations across **COCO-20ⁱ** and **PASCAL-Part** under both **1-shot** and **5-shot** protocols (**87,213 test episodes** across 14 full 4-fold evaluations).

### 1. COCO-20ⁱ Official Test Benchmark (4 Folds, 4,000 Episodes)

| Method | Regimen | Folds | Mean mIoU (%) | Per-Fold Breakdown [0, 1, 2, 3] | $\Delta$ vs Baseline |
|---|:---:|:---:|:---:|---|:---:|
| **Baseline (Official INSID3)** | 1-shot | 4/4 | **56.13%** | 54.9, 60.4, 54.4, 54.8 | — |
| **No Debiasing ($s=0$)** | 1-shot | 4/4 | **53.86%** | 51.7, 57.3, 53.5, 52.8 | **$-2.27\%$** |
| **Task 1 Selected (`sel_task1`)**<br>*(Spectral $k=500, \gamma=4$)* | 1-shot | 4/4 | **56.13%** | 54.5, 60.4, 54.9, 54.7 | $\mathbf{-0.00\%}$ |
| **Task 2 Selected (`sel_task2`)**<br>*(Multi-Prototype LSE $M=2$)* | 1-shot | 4/4 | **55.95%** | 54.8, 59.6, 54.6, 54.8 | $-0.19\%$ |
| **Combined Selected (`sel_combined`)** | 1-shot | 4/4 | **55.99%** | 54.6, 60.0, 54.8, 54.6 | $-0.15\%$ |
| **Baseline (Official INSID3)** | 5-shot | 4/4 | **64.70%** | 64.3, 66.9, 64.8, 62.9 | — |
| **Combined Selected (`sel_combined`)** | 5-shot | 4/4 | **64.23%** | 63.2, 66.8, 64.3, 62.6 | $-0.48\%$ |

---

### 2. PASCAL-Part Official Test Benchmark (4 Folds, 8,459 Episodes)

| Method | Regimen | Folds | Mean mIoU (%) | Per-Fold Breakdown [0, 1, 2, 3] | $\Delta$ vs Baseline |
|---|:---:|:---:|:---:|---|:---:|
| **Baseline (Official INSID3)** | 1-shot | 4/4 | **49.13%** | 45.4, 53.2, 44.9, 52.9 | — |
| **No Debiasing ($s=0$)** | 1-shot | 4/4 | **48.07%** | 43.8, 54.7, 42.8, 51.0 | **$-1.05\%$** |
| **Task 1 Selected (`sel_task1`)**<br>*(Spectral $k=500, \gamma=4$)* | 1-shot | 4/4 | **49.05%** | 45.4, 52.8, 44.9, 53.0 | $-0.07\%$ |
| **Task 2 Selected (`sel_task2`)**<br>*(Multi-Prototype LSE $M=2$)* | 1-shot | 4/4 | **48.51%** | 45.2, 51.7, 44.3, 52.8 | $-0.62\%$ |
| **Combined Selected (`sel_combined`)** | 1-shot | 4/4 | **48.55%** | 45.3, 51.8, 44.3, 52.8 | $-0.58\%$ |
| **Baseline (Official INSID3)** | 5-shot | 4/4 | **57.58%** | 51.1, 67.0, 51.6, 60.6 | — |
| **Combined Selected (`sel_combined`)** | 5-shot | 4/4 | **55.73%** | 51.0, 61.8, 50.2, 59.9 | $-1.85\%$ |

---

## 10. Chapter 9: Honest Result Analysis — What Does It All Mean?

1. **Positional Debiasing is Universally Essential:**  
   Across all 4 test folds of COCO-20ⁱ and PASCAL-Part, omitting debiasing hurts performance significantly (**$-2.27\%$ on COCO**, **$-1.05\%$ on PASCAL-Part**, both $p = 0.000$).
2. **Task 1 (Adaptive Spectral Debiasing) Matches Test Baselines & Gains on Tuning/Small Objects:**  
   On unseen test folds, `sel_task1` achieves **56.13% on COCO** ($-0.00\%$, $p=0.977$) and **49.05% on PASCAL-Part** ($-0.07\%$, $p=0.399$). On tuning episodes it improves $+0.44\%$ ($p=0.040$), led by $+0.98\%$ on the small-object tercile, and on SPair-71k semantic correspondence `token_ls` reaches $+0.29\%$ point PCK@0.10 over hard projection.
3. **Why Multi-Prototype ($M \ge 2$) Underperformed on Test Folds:**  
   In 1-shot test folds, `sel_task2` scored 55.95% on COCO ($-0.19\%$, $p=0.312$) and 48.51% on PASCAL-Part ($-0.62\%$, $p=0.000$). In 5-shot, the single global average acts as a strong noise filter across support shots (reaching 64.70% / 57.58%), whereas preserving shot-specific sub-clusters in `sel_combined` slightly increases background spillover (64.23% / 55.73%, $p=0.022$ / $p=0.000$).

---

## 11. Chapter 10: Summary Tables & Numerical Proof

### Table 1: Baseline Reproduction & Sanity Checks

| Experiment | Split | Description | mIoU (%) | FB-IoU (%) | $\Delta$ vs Baseline | $p$-value |
|---|:---:|---|:---:|:---:|:---:|:---:|
| **`baseline`** | COCO-tune | Official INSID3 ($s=500$, $M=1$) | **57.48%** | **76.74%** | $0.00\%$ | — |
| **`t1_none`** | COCO-tune | Zero debiasing ($s=0$) | **53.13%** | **74.23%** | $-4.34\%$ | $0.000$ |
| **`t2_check_m1`** | COCO-tune | Multi-prototype code with $M=1$ | **57.48%** | **76.74%** | $0.00\%$ | $1.000$ |
| **`baseline`** | COCO-test | Full 4-fold test evaluation | **56.13%** | **76.02%** | $0.00\%$ | — |
| **`baseline_no_debias`** | COCO-test | Full 4-fold test without debiasing | **53.86%** | **74.80%** | $-2.27\%$ | $0.000$ |

---

### Table 2: SPair-71k Semantic Correspondence (12,234 Image Pairs, ViT-B 768)

| Method | PCK@0.05 | PCK@0.10 | PCK@0.15 | PCK@0.20 | $\Delta$@0.10 vs Hard |
|---|:---:|:---:|:---:|:---:|:---:|
| **Original (No Debiasing)** | 30.08% | 46.85% | 55.62% | 61.26% | $-5.72\%$ |
| **Hard Debiasing (Baseline)** | 33.68% | 52.57% | 62.51% | 68.73% | $0.00\%$ |
| **Spectral Attenuation ($k=20, \gamma=1$)** | 33.66% | 52.54% | 62.56% | 68.80% | $-0.03\%$ |
| **Calibrated MMSE ($\beta=1.0$)** | 33.83% | **52.73%** | **62.61%** | **68.81%** | **+0.16%** |
| **Token-wise Debiasing (`least_squares`)** | **34.21%** | **52.86%** | 62.38% | 68.36% | **+0.29%** |

---

### Table 3: Latency & GPU Resource Profile (Measured on RTX 3070 8 GB)

| Model / Strategy | Average Runtime | Peak GPU Memory |
|---|:---:|:---:|
| **Baseline INSID3 (1-shot)** | 546 ms / episode | 1,562 MB |
| **Task 1 Spectral Debiasing (1-shot)** | 548 ms / episode | 1,564 MB |
| **Task 1 Calibrated Debiasing (1-shot)** | 547 ms / episode | 1,564 MB |
| **Task 2 Multi-Prototype ($M=2$ LSE, 1-shot)** | 546 ms / episode | 1,548 MB |
| **Combined Selected (1-shot)** | 548 ms / episode | 1,550 MB |
| **Baseline INSID3 (5-shot)** | 1,190 ms / episode | 2,258 MB |
| **Combined Selected (5-shot)** | 1,202 ms / episode | 2,260 MB |

---

### Final Summary
* Verified all 40 upstream files byte-for-byte against `visinf/INSID3@0c165a1` and passed all 126 unit tests.
* Executed **140 completed runs** (77 tuning runs, 56 test-fold runs across 14 benchmarks, and 7 SPair-71k evaluations) spanning **164,213 segmentation episodes and 85,638 correspondence image pairs**.
* Demonstrated that DINOv3 positional debiasing is indispensable across benchmarks (**$-2.27\%$ COCO test**, **$-1.05\%$ PASCAL-Part test**, **$-4.34\%$ COCO tune**).
* Showed that smooth **Spectral Wiener Debiasing** improves small-object segmentation and tuning mIoU (**$+0.44\%, p=0.040$**) while matching hard projection on full test folds, **Token-wise Debiasing** leads SPair-71k correspondence (**$+0.29\%$ point PCK@0.10**), and naive high-$M$ **Multi-Prototype Matching** degrades under fixed thresholding due to extreme-value score inflation.
