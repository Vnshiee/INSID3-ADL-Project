# INSID3 Research: Adaptive Positional Debiasing and Multi-Prototype Matching

> **Revision (2026-11-10):** a second round of experiments re-examines the first-round results with
> control experiments, 4,000 tuning episodes and paired statistics (`run_revision.sh`, guide:
> [`docs/RUN_GUIDE.md`](docs/RUN_GUIDE.md)). Results and conclusions will be added once it is complete. Literature notes: [`docs/READING_NOTES.md`](docs/READING_NOTES.md).
> Code explained from the basics: [`docs/CODE_WALKTHROUGH.md`](docs/CODE_WALKTHROUGH.md).
>
> **Compute note.** Code development and the first experimental round ran on an RTX 3070 (8 GB). The
> second-round benchmarks were computed on a larger GPU server for speed. Every run records its hardware
> in `env.json`. Methods are only ever compared with a baseline computed on the same machine, because
> different GPUs give slightly different floating-point results.


[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.7.1](https://img.shields.io/badge/pytorch-2.7.1-orange.svg)](https://pytorch.org/)
[![Tests Passing](https://img.shields.io/badge/tests-152%2F152%20passed-brightgreen.svg)](tests/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

This repository presents an independent research extension of **INSID3: Training-Free In-Context Segmentation with DINOv3** (Cuttano et al., CVPR 2026, [official repo](https://github.com/visinf/INSID3)).

The project investigates two extensions of the INSID3 pipeline (assignment: [`topic.md`](topic.md)):
1. **Adaptive Positional Debiasing (Task 1):** Generalising the hard orthogonal projection to a parametric family $\tilde{\mathbf{x}} = \text{normalize}(\mathbf{x} - \mathbf{U}(\boldsymbol{\lambda} \odot \mathbf{U}^\top \mathbf{x}))$. Five adaptive schemes are implemented and evaluated: **Uniformly Weighted**, **Spectral Wiener Attenuation**, **Calibrated MMSE**, **Image-Dependent**, and **Token-Dependent** debiasing.
2. **Multi-Prototype Reference Matching (Task 2):** Representing the reference region by several clustered prototypes instead of the single mean prototype (via Spherical $k$-Means and Hierarchical Agglomerative clustering), paired with flexible many-to-many and soft aggregation rules (**Max**, **Weighted Area**, **Log-Sum-Exp**, **Symmetric Chamfer**, and **Unbalanced Optimal Transport**).

---

## 📖 Documentation and Results

* **[`docs/`](docs/)** — methodology, mathematical formulation, implementation guide, code walkthrough, experimental protocol.
* **[`results/`](results/)** — machine-generated tables of the first round (`summary.md`, `comparisons.md`, `selection_report.md`, calibration figures).
* **[`configs/selected/`](configs/selected/)** — configurations selected on the COCO tuning split before any test run.

Results and conclusions of both rounds will be summarised here when the second round is complete.

---

## 📂 Repository Structure

```
.
├── topic.md                  # Formal problem definition & task specifications
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
├── tests/                    # Unit & regression test suite (152 passing tests)
│   ├── test_baseline_equivalence.py # Mathematical equivalence guarantees
│   ├── test_debiasing.py     # Debiasing edge cases & boundary invariants
│   ├── test_prototypes.py    # Spherical k-means & agglomerative clustering tests
│   ├── test_matching.py      # Aggregation ordering & OT limit tests
│   └── ...                   # CLI, config, and BF16/FP32 precision tests
│
├── docs/                     # Comprehensive architectural documentation
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

# Run unit test suite (152 tests)
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
