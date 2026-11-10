# INSID3 Research: Adaptive Positional Debiasing and Multi-Prototype Matching

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.7.1](https://img.shields.io/badge/pytorch-2.7.1-orange.svg)](https://pytorch.org/)
[![Tests Passing](https://img.shields.io/badge/tests-126%2F126%20passed-brightgreen.svg)](tests/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Status: Experiments Completed](https://img.shields.io/badge/experiments-completed%20%26%20verified-success.svg)](results/)

This repository presents an independent research extension of **INSID3: Training-Free In-Context Segmentation with DINOv3** (Cuttano et al., CVPR 2026, [official repo](https://github.com/visinf/INSID3)).

The project addresses two fundamental bottlenecks in training-free in-context visual segmentation:
1. **Adaptive Positional Debiasing (Task 1):** Replacing the authors' heuristic hard orthogonal projection with a generalized parametric family $\tilde{\mathbf{x}} = \text{normalize}(\mathbf{x} - \mathbf{U}(\boldsymbol{\lambda} \odot \mathbf{U}^\top \mathbf{x}))$. We introduce and benchmark five adaptive schemes: **Uniformly Weighted**, **Spectral Wiener Attenuation**, **Calibrated MMSE**, **Image-Dependent**, and **Token-Dependent** debiasing.
2. **Multi-Prototype Reference Matching (Task 2):** Replacing the single reference prototype (global mean token) with clustered sub-object representations (via Spherical $k$-Means and Hierarchical Agglomerative clustering), paired with flexible many-to-many and soft aggregation rules (**Max**, **Weighted Area**, **Log-Sum-Exp**, **Symmetric Chamfer**, and **Unbalanced Optimal Transport**).

---

## 📖 Key Reports & Documentation

* **[`WhatWeDid.md`](WhatWeDid.md)** — **Complete End-to-End Journey & Detailed Analysis**: A comprehensive, story-driven, non-hype report documenting our entire journey from start to finish, including all empirical tables, statistical tests ($p$-values), failure mode analyses, and negative findings.
* **[`docs/RESULTS.md`](docs/RESULTS.md)** — Master empirical results across COCO `tune`, COCO-20ⁱ test, PASCAL-Part test, and SPair-71k benchmarks, including paired bootstrap confidence intervals.
* **[`results/`](results/)** — Raw machine-generated CSV/JSON/Markdown outputs, paired bootstrap comparisons, episode failure CSVs, and calibration spectra.
* **[`configs/selected/`](configs/selected/)** — Configurations selected strictly on the `coco-tune` split without touching test folds.

---

## 🔬 Empirical Highlights & Key Findings

All experiments were executed on an **NVIDIA GeForce RTX 3070 GPU (8 GB VRAM)** under PyTorch 2.7.1 + CUDA 12.6 across frozen DINOv3 ViT-B/16 (768d) and ViT-L/16 (1024d) backbones with strict RNG isolation and episode fingerprint verification. Values match [`results/summary.md`](results/summary.md) and [`results/comparisons.md`](results/comparisons.md).

### 1. Task 1: Adaptive Positional Debiasing (1,000 Paired COCO Tuning Episodes)

| Method / Configuration | Description | mIoU (%) | $\Delta$ vs Baseline | 95% Confidence Interval | $p$-value |
|---|---|:---:|:---:|:---:|:---:|
| **`t1_none`** | Zero debiasing ($s=0$) | 53.13% | $-4.34\%$ | $[-5.91, -2.81]$ | $0.000$ |
| **`baseline` (Official)** | Hard projection ($s=500$) | 57.48% | $0.00\%$ | — | — |
| **`t1_calibrated_b1.0`** | Empirical MMSE gain vector ($\beta=1.0$) | 56.87% | $-0.61\%$ | $[-1.77, +0.40]$ | $0.216$ |
| **`t1_token_ls_b1.0`** | Spatial template subtraction (`least_squares`) | 57.06% | $-0.41\%$ | $[-1.53, +0.68]$ | $0.460$ |
| **`t1_spectral_k500_g1`** | Smooth Wiener attenuation ($\gamma=1$) | 57.84% | $+0.37\%$ | $[-0.22, +1.06]$ | $0.204$ |
| **`t1_spectral_k500_g4`** ⭐ | **Steep Wiener attenuation ($\gamma=4$)** | **57.92%** | $\mathbf{+0.44\%}$ | $[\mathbf{+0.02}, \mathbf{+1.02}]$ | $\mathbf{0.040}$ |

* **Statistical Proof on Tuning Split:** Spectral Wiener attenuation (`t1_spectral_k500_g4`) achieves **$+0.44\%$ mIoU** ($p = 0.040 < 0.05$). In episode-level failure analysis ([`results/fail_spectral_k500_g4.csv`](results/fail_spectral_k500_g4.csv)), the largest improvement comes from the **small GT-area tercile** (`gt_area <= 26,530` px at $1024^2$, $n=334$), where mean foreground IoU improves by **$+0.98\%$** ($48.09\%$ vs $47.11\%$).
* **Semantic Correspondence (SPair-71k, 12,234 test pairs):** Token-level subtraction (`token_ls`: `52.86%`) and calibrated MMSE debiasing (`calibrated_b1`: `52.73%`) outperform the hard projection baseline (`52.57%` PCK@0.10; un-debiased features score `46.85%`).

### 2. Task 2: Multi-Prototype Reference Matching (COCO Tuning)

| Aggregation Rule | $M=1$ | $M=2$ | $M=3$ | $M=4$ | $M=6$ | $M=8$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Max Aggregation** | 57.48% | 57.25% | 56.39% | 55.94% | 55.79% | 55.82% |
| **Weighted Area** | 57.48% | 57.20% | 57.03% | 56.64% | 56.50% | 56.31% |
| **Log-Sum-Exp ($T=0.05$)** ⭐ | 57.48% | **57.56%** | 57.19% | 56.69% | 56.48% | 56.38% |

* **Empirical Discovery:** Increasing prototype count $M$ with the fixed downstream merge threshold $\alpha=0.20$ increases false-positive background bleed because $\max_{j=1}^M \langle x_i, q_j \rangle$ shifts similarity scores upward; naively raising $\alpha \in \{0.25, 0.30, 0.35\}$ hurts recall (`54.02%`, `52.17%`, `49.20%` for `max` $M=4$).
* **Smooth Softmax Wins:** Log-Sum-Exp aggregation ($M=2, T=0.05$) reaches **57.56%** ($+0.08\%$, $p=0.951$) and improves the **medium GT-area tercile** (`26,530 < gt_area <= 101,262` px, $n=333$) by **$+0.60\%$** ($60.57\%$ vs $59.97\%$).
* **Many-to-Many Matching:** Unbalanced Optimal Transport Sinkhorn matching ($\epsilon=0.02, \rho=0.1$) achieves **57.32%**, outperforming raw Symmetric Chamfer ($56.95\%$) and `max` ($55.94\%$) at $M=4$.

### 3. Full 4-Fold Test Benchmark Generalization

| Model Configuration | Dataset & Regime | Folds | Overall mIoU | Per-Fold mIoU [F0, F1, F2, F3] | $\Delta$ vs Baseline ($p$) |
|---|---|:---:|:---:|---|:---:|
| **Baseline (Official INSID3)** | COCO-20ⁱ 1-shot (4,000 eps) | 4/4 | **56.13%** | 54.9%, 60.4%, 54.4%, 54.8% | — |
| **No Debiasing (`baseline_no_debias`)** | COCO-20ⁱ 1-shot (4,000 eps) | 4/4 | 53.86% | 51.7%, 57.3%, 53.5%, 52.8% | $-2.27\%$ ($p=0.000$) |
| **Task 1 Selected (`sel_task1`)** | COCO-20ⁱ 1-shot (4,000 eps) | 4/4 | **56.13%** | 54.5%, 60.4%, 54.9%, 54.7% | $-0.00\%$ ($p=0.977$) |
| **Task 2 Selected (`sel_task2`)** | COCO-20ⁱ 1-shot (4,000 eps) | 4/4 | 55.95% | 54.8%, 59.6%, 54.6%, 54.8% | $-0.19\%$ ($p=0.312$) |
| **Combined Selected (`sel_combined`)** | COCO-20ⁱ 1-shot (4,000 eps) | 4/4 | 55.99% | 54.6%, 60.0%, 54.8%, 54.6% | $-0.15\%$ ($p=0.475$) |
| **Baseline (Official INSID3)** | COCO-20ⁱ 5-shot (4,000 eps) | 4/4 | **64.70%** | 64.3%, 66.9%, 64.8%, 62.9% | — |
| **Combined Selected (`sel_combined`)** | COCO-20ⁱ 5-shot (4,000 eps) | 4/4 | 64.23% | 63.2%, 66.8%, 64.3%, 62.6% | $-0.48\%$ ($p=0.022$) |
| **Baseline (Official INSID3)** | PASCAL-Part 1-shot (8,459 eps) | 4/4 | **49.13%** | 45.4%, 53.2%, 44.9%, 52.9% | — |
| **No Debiasing (`baseline_no_debias`)** | PASCAL-Part 1-shot (8,459 eps) | 4/4 | 48.07% | 43.8%, 54.7%, 42.8%, 51.0% | $-1.05\%$ ($p=0.000$) |
| **Task 1 Selected (`sel_task1`)** | PASCAL-Part 1-shot (8,459 eps) | 4/4 | 49.05% | 45.4%, 52.8%, 44.9%, 53.0% | $-0.07\%$ ($p=0.399$) |
| **Task 2 Selected (`sel_task2`)** | PASCAL-Part 1-shot (8,459 eps) | 4/4 | 48.51% | 45.2%, 51.7%, 44.3%, 52.8% | $-0.62\%$ ($p=0.000$) |
| **Combined Selected (`sel_combined`)** | PASCAL-Part 1-shot (8,459 eps) | 4/4 | 48.55% | 45.3%, 51.8%, 44.3%, 52.8% | $-0.58\%$ ($p=0.000$) |
| **Baseline (Official INSID3)** | PASCAL-Part 5-shot (8,459 eps) | 4/4 | **57.58%** | 51.1%, 67.0%, 51.6%, 60.6% | — |
| **Combined Selected (`sel_combined`)** | PASCAL-Part 5-shot (8,459 eps) | 4/4 | 55.73% | 51.0%, 61.8%, 50.2%, 59.9% | $-1.85\%$ ($p=0.000$) |

---

## 📂 Repository Structure

```
.
├── WhatWeDid.md              # Detailed journey, story-driven report & analysis
├── topic.md                  # Formal problem definition & task specifications
├── PROJECT_STATE.md          # Verification log & component execution states
│
├── insid3_ext/               # Novel research extension library
│   ├── debiasing.py          # Parametric debiasing (weighted, spectral, calibrated, token, image)
│   ├── prototypes.py         # Multi-prototype builders (spherical k-means, agglomerative)
│   ├── matching.py           # Similarity rules (max, weighted, LSE, Chamfer, unbalanced OT)
│   ├── calibration.py        # Empirical basis calibration & MMSE gain computation
│   ├── model.py              # INSID3Ext model wrapper & forward overrides
│   ├── experiment.py         # Resumable episode runner with SHA256 fingerprint verification
│   ├── builder.py            # Model factory & configuration wiring
│   └── runtime.py            # Device, seed, and environment management
│
├── configs/                  # Experiment configuration files
│   ├── base.yaml             # Shared defaults (model, data, debias, prototypes)
│   ├── baseline.yaml         # Official INSID3 baseline configuration
│   ├── baseline_no_debias.yaml # Baseline without positional debiasing
│   ├── baseline_crf.yaml     # Baseline with DenseCRF post-processing
│   ├── task1/                # 5 Task 1 base configs (swept across 29 runs)
│   ├── task2/                # 7 Task 2 base configs (swept across 42 runs)
│   ├── combined/             # Joint Task 1 + Task 2 experiment configs
│   ├── correspondence/       # SPair-71k keypoint transfer configs
│   └── selected/             # Auto-selected best models (task1, task2, combined)
│
├── results/                  # Compiled empirical outputs & figures
│   ├── summary.md            # Fold-averaged segmentation & SPair-71k summary tables
│   ├── summary_segmentation_runs.csv # Raw metrics for all 133 segmentation run-folds
│   ├── comparisons.md        # Paired bootstrap statistical tests & confidence intervals
│   ├── comparisons.json      # Structured statistical comparison outputs
│   ├── selection_report.md   # Objective validation-split selection results
│   ├── fail_*.csv            # Episode-level failure analysis breakdowns
│   └── calibration/          # Empirical spectra & token figures + JSON summaries
│
├── scripts/                  # Command-line tools and utilities
│   ├── run_segmentation.py   # Main evaluation runner for 1-shot / 5-shot segmentation
│   ├── run_correspondence.py # SPair-71k PCK semantic keypoint evaluation runner
│   ├── calibrate_positional.py # Offline empirical calibration script
│   ├── select_best.py        # Automated validation model selector
│   ├── aggregate_results.py  # Collect metrics.json into summary.md and CSV
│   ├── compare_all.py        # Paired bootstrap significance testing vs baseline
│   ├── compare_runs.py       # Paired bootstrap between any two run directories
│   ├── analyze_failures.py   # Size/IoU correlation and failure mode extractor
│   ├── visualize.py          # Calibration curve and spectrum visualizer
│   └── verify_upstream.py    # Byte-by-byte integrity checker for upstream files
│
├── tests/                    # Unit & regression test suite (126 passing tests)
│   ├── test_baseline_equivalence.py # Mathematical equivalence guarantees
│   ├── test_debiasing.py     # Debiasing edge cases & boundary invariants
│   ├── test_prototypes.py    # Spherical k-means & agglomerative clustering tests
│   ├── test_matching.py      # Aggregation ordering & OT limit tests
│   └── ...                   # CLI, config, and BF16/FP32 precision tests
│
├── docs/                     # Comprehensive architectural documentation
│   ├── RESULTS.md            # Master empirical results table
│   ├── METHODOLOGY.md        # Mathematical derivations and architectural decisions
│   ├── REPOSITORY_AUDIT.md   # Audit of official code vs paper formulation
│   ├── FAILURE_ANALYSIS.md   # Systematic categorization of model errors
│   ├── ABLATION_STUDY.md     # Ablation axes and hypotheses
│   └── EXPERIMENTAL_PROTOCOL.md # Data splits, seed isolation, and metrics
│
├── models/                   # Upstream INSID3 model definitions (unmodified)
├── datasets/                 # Upstream dataset loaders (unmodified)
├── utils/                    # Upstream utilities (unmodified)
└── *.sh                      # Bash workflow automation scripts
```

---

## 🚀 Reproduction & Usage

### 1. Setup Environment & Verify Integrity

```bash
# Clone the repository
git clone https://github.com/Vnshiee/INSID3-ADL-Project.git
cd INSID3-ADL-Project

# Set up conda environment
conda create -n insid3 python=3.10 -y
conda activate insid3
pip install -r requirements.txt -r requirements-research.txt

# Verify upstream code integrity (must output 40/40 identical)
python scripts/verify_upstream.py

# Run unit test suite (126 tests)
pytest tests/ -q
```

### 2. Run Pre-Configured Experiments

```bash
# Evaluate baseline model on COCO tuning split
python scripts/run_segmentation.py --config configs/baseline.yaml --set data.split=tune

# Evaluate selected Task 1 Spectral model (Wiener attenuation)
python scripts/run_segmentation.py --config configs/selected/task1.yaml --set data.split=tune

# Evaluate selected Task 2 Multi-Prototype model (LSE M=2)
python scripts/run_segmentation.py --config configs/selected/task2.yaml --set data.split=tune

# Evaluate semantic correspondence on SPair-71k
python scripts/run_correspondence.py --config configs/correspondence/base_readme.yaml \
  --set debias.mode=spectral debias.knee_rank=20 debias.sharpness=1.0
```

### 3. Generate Reports & Statistical Significance

```bash
# Recompute summary CSV and markdown tables
python scripts/aggregate_results.py --root outputs --out results/summary

# Compute paired bootstrap p-values and 95% confidence intervals vs baseline
python scripts/compare_all.py --root outputs --reference baseline --out results/comparisons

# Extract episode-level failure mode CSVs
python scripts/analyze_failures.py \
  --a outputs/baseline/coco-tune/fold0_shot1_seed0 \
  --b outputs/t1_spectral_k500_g4/coco-tune/fold0_shot1_seed0 \
  --out results/fail_spectral_k500_g4
```

---

## ⚖️ License and Attribution

* Original INSID3 implementation © 2026 Claudia Cuttano et al. (Apache-2.0 License).
* Research extensions, experimental evaluation, and documentation © 2026 Vnshiee (Apache-2.0 License). See [`NOTICE`](NOTICE) and [`LICENSE`](LICENSE) for full details.

```bibtex
@inproceedings{cuttano2026insid3,
  title     = {{INSID3}: Training-Free In-Context Segmentation with {DINOv3}},
  author    = {Claudia Cuttano and Gabriele Trivigno and Christoph Reich and Daniel Cremers and Carlo Masone and Stefan Roth},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```
