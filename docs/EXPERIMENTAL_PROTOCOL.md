# Experimental protocol

## 1. Benchmarks (official, unchanged)

| Task | Dataset | Folds | Episodes / fold | Metric |
|---|---|---|---|---|
| semantic | COCO-20ⁱ | 4 | 1000 | class-wise mIoU (fg), averaged over folds |
| semantic | LVIS-92ⁱ | 10 | 2300 | same |
| semantic | ISIC 2018, Chest X-ray, SUIM, iSAID-5ⁱ (3 folds) | 1 / 3 | dataset-defined | same |
| part | PASCAL-Part (4 folds, ≤ 2500), PACO-Part (4 folds, 2500) | 4 | as stated | same |
| personalised | PerMIS | 1 | 2 per video (432) | same |
| correspondence | SPair-71k test | — | all pairs | image-wise PCK@{0.05, 0.10, 0.15, 0.20} |

Evaluation code: official `Evaluator.classify_prediction`, `AverageMeter.compute_iou`, `PCKEvaluator`.
Prediction and ground truth are compared at `image_size²` (official behaviour, REPOSITORY_AUDIT D8).

## 2. Fixed settings (paper Tab. 6, never tuned on test data)

ViT-L/16, 1024 px, τ = 0.6, α = 0.2, s = 500, seed 0, `num_workers = 0`, 1-shot unless stated.
Mask refinement: **bilinear for all ablations**; the final four methods (baseline, Task 1, Task 2,
combined) are additionally evaluated with CRF if the CRF package builds (`CRF=1 ./run_baseline.sh`,
`--set model.mask_refiner=crf`), because paper Tab. 1 numbers use CRF.

## 3. Data separation

| Data | Used for | Images |
|---|---|---|
| Calibration set | ρ̂, templates (Task 1) | 500 random COCO **train2014** images, no labels |
| Tuning episodes | choosing modes and hyper-parameters (Tasks 1, 2, combined) | COCO-20ⁱ `splits/trn` (train2014), all 80 classes, 1000 episodes, seed 0 |
| Test episodes | final numbers only | official splits (val2014 for COCO) |

This mirrors the paper, which selected its hyper-parameters on the COCO-20ⁱ training split (Supp. B).
The overlap between tuning and test images is computed and logged at run time (expected 0). Caveat:
LVIS/PACO use COCO-2017 images, which overlap COCO-2014 train images; calibration uses no labels, so this
is unsupervised feature statistics, not label leakage — reported as a limitation.

## 4. Selection rule (decided before seeing any result)

1. Run every sweep in `run_task1.sh` / `run_task2.sh` on tuning episodes.
2. `scripts/select_best.py` picks the highest tuning mIoU per task (ties → alphabetical).
3. Combined = best Task-1 debias section + best Task-2 prototype section (+ its α); a small combined
   sweep (grouping space, α ∈ {0.15, 0.25}) is also run on tuning episodes.
4. Only the selected configs are evaluated on the test benchmarks. Nothing is re-tuned afterwards.

## 5. Reproducibility

* Seeds: `torch`, `numpy`, `random` = 0; cudnn deterministic; TF32 on Ampere+ (as official).
* Every run directory stores `config.yaml` (fully resolved), `env.json` (versions, GPU, git commit),
  `log.txt`, `episodes.jsonl` (per-episode intersections/unions, fingerprints, time, memory, diagnostics),
  `metrics.json`.
* Episode fingerprints (hash of class id and masks) prove that two methods saw identical episodes.
* Resumable: re-running a command skips finished episodes after re-checking fingerprints.

## 6. Baseline reproduction check (must pass before interpreting anything)

* `t2_check_m1` (Task-2 code path, M = 1) must equal `baseline` on 1-shot tuning episodes (Δ = 0.00).
* `baseline` on COCO-20ⁱ (4 folds) should be close to the published 57.6 (CRF) — bilinear is expected to be
  somewhat lower (exact gap unknown). `baseline_no_debias` should be close to 54.5 (Supp. Tab. 5).
* `corr_base768_original` / `corr_base768_hard` should be close to the README PCK values
  (30.1/46.8/55.6/61.2 and 33.7/52.6/62.5/68.7).
Large deviations indicate an environment problem (weights, resolution, data) — see TROUBLESHOOTING.md.
Expected sources of small deviations: GPU model, cuDNN/TF32 non-determinism, library versions.

## 7. Statistical reporting

* Paired bootstrap over episodes (10 000 resamples for final results, stratified by fold), statistic =
  fold-averaged mIoU difference; report Δ, 95 % CI and two-sided bootstrap p (`compare_runs.py`, `compare_all.py`).
* PCK: paired bootstrap over pairs.
* A method is called "better" only if the CI excludes 0 on the main benchmark(s) of the task; results on all
  benchmarks are reported, including losses. Multiple comparisons: treat tuning-split differences as
  exploratory; only the selected configs are confirmatory.
* Runtime (ms/episode) and peak GPU memory are reported from `episodes.jsonl` (same GPU for all methods).
