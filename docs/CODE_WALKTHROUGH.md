# Code walkthrough — the official INSID3 code first, then every change we made

Read this with the code open. Line numbers refer to the official files (unchanged in this repository).
Our files contain, at every changed spot, the **original lines as a comment** followed by our version
(`# ORIGINAL (models/insid3.py Lxxx):` … `# OURS:`).

---

## Part A — The official INSID3 code

### A.1 What one evaluation run does (`inference_segmentation.py`)

1. `main` (L23): set the random seeds (torch, numpy, random = 0), build the model, call `evaluate`.
2. `evaluate` (L56): build the dataset (`datasets/coco.py` etc.). Each item is one **episode**:
   reference image(s) + reference mask(s) + target image + target ground-truth mask + class id.
   Episodes are drawn at random with numpy's global random generator — so the seed fixes them.
3. For every episode (L65-93): `model.set_reference(...)` for each shot, `model.set_target(...)`,
   `model.segment()` → predicted mask; the ground truth is resized to the prediction size;
   `Evaluator.classify_prediction` (utils/metrics.py) counts intersection and union;
   `AverageMeter` adds them up per class.
4. At the end: **mIoU** = mean over classes of (total intersection / total union) of the foreground.

### A.2 The model (`models/insid3.py`, class `INSID3`)

**Construction** (L21-53): stores the settings (image size 1024, s = 500, τ = 0.6, merge threshold 0.2)
and computes the positional basis once with `_build_positional_basis`.

**`_build_positional_basis`** (L246-261) — "where is position in the features?"
- An all-zero image, ImageNet-normalised (L249-252), is encoded by DINOv3 (L253-255).
- Patch features are scaled to length 1 (L256), arranged as a C × P matrix and the average patch is
  subtracted (L258-259). On an empty image, patches can only differ because of their position.
- SVD (L260); the first s = 500 left singular vectors are the **positional basis** B (L261).
- Note: this runs inside the constructor, while DINOv3 is still in training mode; its RoPE then rescales the
  patch coordinates by a random factor (see REPOSITORY_AUDIT.md §5 item 8 and `model.basis_eval_mode`).

**`set_reference` / `set_target`** (L65-89): load and resize images to 1024 × 1024, masks likewise;
several references are stacked (k-shot).

**`segment`** (L93-104) calls `predict_mask` and clears the stored inputs.

**`predict_mask`** (L122-189) — the whole method:

| Lines | Step | In plain words |
|---|---|---|
| 133-139 | features | DINOv3 features of references + target (`_extract_features`, L233-241, bf16 on GPU), scaled to length 1 |
| 141-142 | | target features kept in their original form (for grouping) |
| 145-147 | debias | `_debias_features` (L263-271): x → normalize((I − BBᵀ)x) — delete the positional part |
| 150-158 | **prototype** | average of the debiased reference features inside the mask (per shot, then over shots), length 1 |
| 162-169 | candidates | `_locate_candidates` (L275-307): **forward** = target patches with ⟨x, prototype⟩ > 0; **backward** = target patches whose most similar reference patch lies inside the reference mask (majority over shots); candidate = both |
| 174-176 | clustering | agglomerative clustering of the target with ORIGINAL features, threshold 1 − τ (utils/clustering.py) → clusters G_1..G_K |
| 178-181 | | each cluster's average debiased feature ("cluster prototype") |
| 184-187 | merge | `_seed_and_aggregate` (L311-364), below |
| 189 | output | `_finalize_mask` (L368-383): upsample the 64 × 64 mask (bilinear or CRF) |

**`_seed_and_aggregate`** (L311-364):
- area weight of each cluster = fraction of its patches that are candidates (L330-334);
- **seed** = candidate cluster whose prototype is most similar to the reference prototype (L337-340);
- intra-image similarity of every cluster to the seed, with ORIGINAL features (L343-347);
- cross-image score of every cluster = mean of ⟨x, prototype⟩ over its patches (L350-354);
- final score = cross × intra × area weight (seed weight = 1) (L357-359);
- keep clusters with final score > 0.2 (L361-363).

**`predict_keypoint`** (L194-228): for SPair-71k — take the (optionally debiased) feature at each
reference keypoint, compare with all target patches, soft-argmax → predicted keypoint.

### A.3 Helpers
- `utils/data.py`: image transform, `downsample_mask` (L102: mask → patch grid; falls back to the
  nearest patch for tiny masks).
- `utils/clustering.py`: `agglomerative_clustering` (scikit-learn, cosine distance, average linkage),
  `compute_cluster_prototypes`.
- `utils/metrics.py`: IoU counting, mIoU, PCK for correspondence.
- `models/__init__.py`: loads DINOv3 through torch.hub.

---

## Part B — What we added (`insid3_ext/`) and how it connects

```
scripts/run_segmentation.py ─► insid3_ext/experiment.py (episode loop = A.1, plus logging/resume)
                                   └─► insid3_ext/builder.py ─► INSID3Ext (insid3_ext/model.py)
                                                                   ├─ debiasing.py   (Task 1)
                                                                   ├─ prototypes.py  (Task 2)
                                                                   └─ matching.py    (Task 2)
configs/*.yaml ─► insid3_ext/config.py (all settings, with the reason for every default)
```

### B.1 `model.py` — the only place where official code is replaced

`INSID3Ext` inherits from `INSID3`. It overrides:

| Official method | Our version | What changed |
|---|---|---|
| `_build_positional_basis` (L246) | same computation via `estimate_positional_svd` | keeps all directions and singular values (needed by the new modes); the returned basis is identical |
| `_debias_features` (L263) | `mode = hard` → calls the official method; otherwise `PositionalDebiaser` | Task 1 |
| `predict_mask` (L122) | copy of the official method | only when Task 2 is on (or `full_diagnostics`): (a) forward filter, (b) seed score, (c) cross score use the prototype SET; all other steps copied unchanged |
| `_locate_candidates` backward part (L291-305) | `_backward_mask` | unchanged copy (split out so the forward part can change) |
| `_seed_and_aggregate` (L311) | `_seed_and_aggregate_scores` | unchanged copy, except that the seed score and cross score are passed in |

With Task 2 off and `mode = hard`, the official methods themselves run.

### B.2 `debiasing.py` — Task 1
`x_debiased = normalize(x − U(λ ⊙ Uᵀx))`; λ_k = fraction of direction k removed. Four kinds:
official (hard), static λ (none, weighted, spectral, calibrated, rho_threshold, random, pca),
per-image λ (image), per-patch pattern (token). Each mode's formula and reason are in the file.

### B.3 `calibration.py` — Task 1 analysis
Measures on 500 real (training) images how much of each direction is position (ρ_k), the positional
pattern per patch position, per-image and per-patch statistics, and the feature mean/covariance.

### B.4 `prototypes.py` and `matching.py` — Task 2
Group the masked reference patches (spherical k-means / agglomerative), one prototype per group
(weights = patch shares); score target patches (max / weighted / lse) and target clusters
(prototype / Chamfer / optimal transport).

### B.5 Everything else
- `config.py`: settings and checks; every default says whether it is the official value or a starting
  value that the tuning sweep varies.
- `experiment.py`: the same episode loop as `inference_segmentation.py`, plus per-episode logs
  (`episodes.jsonl`), resume, and episode fingerprints (proof that two methods saw the same episodes).
- `datasets_ext.py`: the official datasets + the COCO tuning split (train2014 images).
- `builder.py`, `runtime.py`: model loading with configurable paths, seeds, timing.
- `scripts/`: run, calibrate, select, compare (paired bootstrap), analyse, plot.
- `run_revision.sh`: the experiment stages.

---

## Part C — How we know the code is right

| Claim | Test |
|---|---|
| Our basis = official basis (bit-identical) | `tests/test_baseline_equivalence.py::test_positional_basis_identical` |
| Extensions off → official predictions | `test_predict_mask_identical_extensions_off`, `test_keypoint_matching_identical` |
| Task-2 code with M = 1 → official predictions (1 shot), pooled M = 1 → official (any shots) | `test_task2_path_m1_reproduces_baseline_1shot`, `tests/test_revision.py::test_pooled_m1_pipeline_reproduces_official_multishot` |
| Diagnostics path → official predictions | `test_full_diagnostics_path_is_identical` |
| Readability rewrite changed no number | `tests/test_golden.py` (186 stored outputs of every mode, compared exactly) |
| Formulas behave as derived (limits, bounds) | `tests/test_debiasing.py`, `test_matching.py`, `test_calibration.py`, `test_prototypes.py` |
| On real data: Task-2 code with M = 1 = baseline | round-1 result `t2_check_m1` = `baseline` (57.4772 = 57.4772) |
