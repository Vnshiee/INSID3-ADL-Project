# 3. Code walkthrough: the official INSID3 code first, then every change we made

Read this with the code open. Line numbers of official files refer to the official repository; our copies
are byte-identical (`python scripts/verify_upstream.py` checks 40 files against their SHA-256 hashes).
At every place where our code replaces an official line, the original line is quoted as a comment
(`# ORIGINAL (models/insid3.py Lxxx): …`) followed by `# OURS: …`.

---

## Part A — The official code

### A.1 One evaluation run (`inference_segmentation.py`)

1. `main` (L23): seeds for torch, numpy and random = 0, deterministic cuDNN, TF32 on new GPUs. Then the
   model is built and `model.eval()` is called (L43-45).
2. `evaluate` (L56): builds the dataset. Each item is one episode (reference images + masks, target image +
   mask, class id), drawn at random with numpy's global generator, so the seed fixes the episodes.
3. Per episode (L65-93): `set_reference` for every shot, `set_target`, `segment()` gives the predicted mask.
   The ground truth is resized to the prediction size (nearest), `Evaluator.classify_prediction` counts
   intersection and union, and `AverageMeter` adds them up per class.
4. At the end: mIoU = mean over classes of (total intersection / total union) of the foreground.

### A.2 The model (`models/insid3.py`, class `INSID3`)

**Constructor** (L21-53): stores the settings (1024 px, s = 500, τ = 0.6, α = 0.2) and computes the
positional basis once (L44).

**`_build_positional_basis`** (L246-261):
* zero image after ImageNet normalisation (L249-252), encoded by DINOv3 in fp32 (L253-255);
* patch vectors normalised (L256), arranged as C × P, mean patch subtracted (L258-259);
* SVD (L260), first s columns of U returned (L261).
* It runs inside the constructor, *before* `model.eval()`, so DINOv3 is still in training mode
  (see Part B.2).

**`_debias_features`** (L263-271): builds `P = I − B Bᵀ`, applies it to every patch, normalises.

**`predict_mask`** (L122-189):

| Lines | Step | What happens |
|---|---|---|
| 133-139 | features | DINOv3 features of references + target in one batch (`_extract_features`, bf16 on the GPU), unit length |
| 141-142 | | the target's original features are kept (for clustering) |
| 145-147 | debias | `_debias_features` |
| 150-158 | prototype | per shot: mean of the debiased features inside the mask; mean over shots; unit length |
| 162-169 | candidates | `_locate_candidates` (L275-307): forward ⟨x, p⟩ > 0 (top 10 % if none), backward: nearest reference patch inside the reference mask (majority over shots) |
| 170-171 | | no candidate → empty mask |
| 174-176 | clustering | agglomerative clustering of the target with ORIGINAL features, distance threshold 1 − τ |
| 178-181 | | unit-length mean of the debiased features of every cluster |
| 184-187 | merging | `_seed_and_aggregate` (L311-364), below |
| 189 | output | `_finalize_mask` (L368-383): bilinear upsampling (or CRF) |

**`_seed_and_aggregate`** (L311-364):
* area weight = share of each cluster's patches that are candidates (L330-334);
* seed = candidate cluster most similar to the prototype (L337-340);
* similarity of every cluster to the seed, with ORIGINAL features (L343-347);
* cross score of every cluster = mean of ⟨x, p⟩ over its patches (L350-354);
* final score = cross × intra × area weight, seed weight forced to 1 (L357-359);
* keep the clusters with final score > 0.2 (L361-363).

**`predict_keypoint`** (L194-228): the (optionally debiased) feature at each source keypoint is compared
with all target patches; a Gaussian-weighted soft-argmax gives the predicted point.

### A.3 Helpers

* `utils/data.py`: image transform; `downsample_mask` (L102) shrinks a mask to the patch grid (bilinear >
  0.5, then nearest, then the centre patch, so every non-empty mask keeps at least one patch).
* `utils/clustering.py`: `agglomerative_clustering` (scikit-learn, precomputed cosine distance, average
  linkage) and `compute_cluster_prototypes`.
* `utils/metrics.py`: IoU counting, `AverageMeter` (mIoU, FB-IoU), `PCKEvaluator`.
* `models/__init__.py`: loads DINOv3 through torch.hub.

---

## Part B — Our code (`insid3_ext/`) and how it plugs in

```
scripts/run_segmentation.py ─► insid3_ext/experiment.py  (episode loop = A.1, plus logs and resume)
                                  └─► insid3_ext/builder.py ─► INSID3Ext  (insid3_ext/model.py)
                                                                 ├─ debiasing.py   (Task 1)
                                                                 ├─ prototypes.py  (Task 2)
                                                                 └─ matching.py    (Task 2)
configs/*.yaml ─► insid3_ext/config.py  (every setting, with the reason for its default)
```

### B.1 `model.py` — the only place where official methods are replaced

`INSID3Ext` *inherits* from `INSID3`. Python then calls our version of a method wherever the official code
calls `self.<method>`, without editing the official file.

| Official method | Ours | What changed |
|---|---|---|
| `_build_positional_basis` (L246) | same computation via `estimate_positional_svd` | keeps all directions and singular values; the returned basis is bit-identical; optional eval-mode basis |
| `_debias_features` (L263) | mode `hard` → `super()._debias_features` (the official code itself); other modes → `PositionalDebiaser` | Task 1 |
| `predict_mask` (L122) | copy of the official method | only when Task 2 is on (or diagnostics are on): the three uses of the prototype change; every other block is copied and marked |
| backward part of `_locate_candidates` (L291-305) | `_backward_mask` | copied unchanged (split out so the forward part can change) |
| `_seed_and_aggregate` (L311) | `_seed_and_aggregate_scores` | copied unchanged, except that the seed score and the cross score come from outside |

With Task 2 off and mode `hard`, the official methods run themselves.

### B.2 `debiasing.py` — Task 1

* `estimate_positional_svd`: the official basis computation line by line, but it keeps U, S and the empty
  image's coordinates.
* `weighted_gains`, `spectral_gains`, `calibrated_gains`: the λ vectors (docs/2_method.md 2.1).
* `PositionalDebiaser.from_config`: chooses the directions and the λ (or the pattern) of the mode, in four
  numbered steps.
* `PositionalDebiaser.__call__`: `c = Uᵀ x`, `removed = λ ⊙ c` (or the pattern), `x − U removed`, normalise.
* **Training-mode basis (our finding).** The official constructor runs before `model.eval()`. In training
  mode DINOv3's RoPE multiplies the patch coordinates by a random factor in [1/2, 2], so the official basis
  comes from one random forward pass, fixed only by the seed. Therefore every script sets the seed *before*
  building the model, and `model.basis_eval_mode=true` can compute the basis in eval mode instead.

### B.3 `calibration.py` — measuring the positional components

`PositionalStatsAccumulator` adds up sums over many images (in float64), and `finalize` turns them into the
template, the positional energy (with the noise correction), the share ρ, per-image and per-position
statistics, and the mean and covariance of real features (for the `pca` and `center` controls).

### B.4 `prototypes.py` and `matching.py` — Task 2

* `spherical_kmeans`: k-means++ start, then alternate "assign to the most similar centre" and "new centre =
  unit-length weighted mean"; its own random generator.
* `build_reference_prototypes`: groups per shot (or pooled), one prototype and one weight per group, plus
  the original prototype for comparison.
* `matching.py`: `aggregate_similarities` (max / weighted / lse) and three region scores. Each region score
  is a plain loop over the target clusters, like the official cross score: `mean_per_cluster`,
  `region_scores_chamfer`, and `region_scores_ot` (Sinkhorn for one cluster at a time in
  `sinkhorn_plan_one_region`).

### B.5 Running experiments

* `config.py`: all settings as dataclasses. A YAML file can inherit another (`_base_`), use environment
  variables (`${env:VAR,default}`), and be overridden from the command line (`--set a.b=value`). Wrong keys
  or values stop the program before any GPU work.
* `experiment.py`: the same episode loop as the official script, plus one JSON line per episode
  (`episodes.jsonl`), resume after an interruption, and an episode fingerprint (a hash of the class and the
  masks) that proves two methods saw the same episodes.
* `datasets_ext.py`: the official datasets plus the COCO tuning split (train2014 images, never in a test fold).
* `builder.py` / `runtime.py`: loading DINOv3 with configurable paths, seeding exactly like the official
  script, timing, memory.
* `scripts/`: run, calibrate, measure bias, select, compare (paired bootstrap), aggregate, analyse failures,
  plot, check environment, check consistency, verify official files.
* `run_experiments.sh`: every experiment of both rounds, as stages.

---

## Part C — How we know the code is right (153 tests, `python -m pytest tests -q`)

The tests run on the CPU with a tiny fake encoder (`tests/conftest.py`) that has the same interface as
DINOv3 and a planted positional pattern.

| Claim | Test |
|---|---|
| our basis = official basis (bit-identical) | `test_baseline_equivalence.py::test_positional_basis_identical` |
| everything off → official predictions and keypoints | `test_predict_mask_identical_extensions_off`, `test_keypoint_matching_identical` |
| Task-2 code with M = 1 → official (1 shot); pooled M = 1 → official (any shots) | `test_task2_path_m1_reproduces_baseline_1shot`, `test_revision.py::test_pooled_m1_pipeline_reproduces_official_multishot` |
| diagnostics path → official predictions | `test_full_diagnostics_path_is_identical` |
| simplifying the code changed no number | `test_golden.py`: 186 stored outputs of every mode, made with the earlier code version |
| formulas behave as derived (limits, bounds, orderings) | `test_debiasing.py`, `test_matching.py`, `test_calibration.py`, `test_prototypes.py` |
| our code never touches the global random generators | `test_kmeans_deterministic_and_rng_isolated`, `test_methods_see_identical_episodes` |
| on real data: Task-2 code with M = 1 = baseline | round-1 run `t2_check_m1` = `baseline` (57.4772 = 57.4772) |
