# Repository audit — official INSID3

**Source of truth:** `github.com/visinf/INSID3`, commit `0c165a10cf52ab91f335883d06260de86854adbe`
(2026-06-26), Apache-2.0, plus the paper (arXiv 2603.28480v1, CVPR 2026, main text + supplementary).
Both were read in full for the parts below. Line numbers refer to the vendored files in this repository,
which are byte-identical to upstream (`python scripts/verify_upstream.py` checks 40 files against
`upstream/UPSTREAM_MANIFEST.sha256`). The official README was moved to `upstream/UPSTREAM_README.md`.

## 1. File inventory (official)

| File | Role |
|---|---|
| `models/insid3.py` | The whole method: `INSID3(nn.Module)` |
| `models/__init__.py` | `build_insid3`, `build_insid3_from_args`; DINOv3 via `torch.hub.load("facebookresearch/dinov3", …, weights="pretrain/…")` |
| `utils/clustering.py` | `agglomerative_clustering` (sklearn, cosine distance, average linkage, threshold `1-τ`), `compute_cluster_prototypes` |
| `utils/data.py` | `build_transform` (square resize to `image_size`, ImageNet norm), `load_image`, `load_mask`, `downsample_mask` |
| `utils/refinement.py` | `upsample_mask` (bilinear > 0.5), GPU dense CRF (`CRF` package, optional) |
| `utils/keypoints.py` | Gaussian-suppressed soft-argmax for keypoint transfer |
| `utils/metrics.py` | `AverageMeter` (class-wise mIoU, FB-IoU), `Evaluator.classify_prediction`, `PCKEvaluator` |
| `utils/visualization.py` | Overlay figures for segmentation / matching |
| `datasets/*.py` | 9 segmentation benchmarks + SPair-71k; episode sampling with the **global numpy RNG** |
| `inference_segmentation.py` | Evaluation loop, mIoU |
| `inference_keypoint_matching.py` | SPair-71k PCK evaluation (`--debiased`) |
| `opts.py` | CLI defaults: large, 1024 px, s=500, τ=0.6, merge 0.2, seed 0, workers 0 |

## 2. Execution trace of `predict_mask` (ViT-L/16 at 1024 px: C = 1024, h = w = 64, P = 4096)

| # | Step | Code | Tensor shapes | Operation |
|---|---|---|---|---|
| 0 | Positional basis (once, at construction) | `_build_positional_basis` L246–261 | image (1,3,1024,1024) → F (1,C,64,64) → E (C,P) → B (C,s) | zeros image after ImageNet normalisation; L2-normalise tokens; subtract token mean; `U,_,_ = svd(E)`; `B = U[:, :s]`; **no autocast** |
| 1 | Feature extraction | `_extract_features` L233–241 | imgs (1,S+1,3,H,W) → (1,S+1,C,h,w) | last layer of DINOv3 (`get_intermediate_layers(n=1, reshape=True)`), **bf16 autocast on CUDA** |
| 2 | Normalisation | L138 | same | L2 over C |
| 3 | Debiasing | `_debias_features` L263–271 | (B·T,C,P) | `normalize((I − BBᵀ) x)` |
| 4 | Reference prototype | L150–158 | fg (C,N_m) → p̃ (C,1) | `normalize(mean_m mean_j fg_m)` on **debiased** features; mask downsampled by `downsample_mask` (bilinear > 0.5, nearest fallback, centre-pixel fallback) |
| 5 | Dense cross similarity | L162–166 | (1,h,w,h,w) per shot | `einsum('bchw,bcxy->bhwxy')` debiased ref × debiased target |
| 6 | Forward mask | `_locate_candidates` L286–289 | (h,w) | `⟨x̃_t, p̃⟩ > 0`; if empty, top-10 % quantile |
| 7 | Backward NN | L292–305 | (h,w) | per target token: argmax over reference tokens; vote if NN inside downsampled ref mask; keep if votes ≥ ⌈S/2⌉ |
| 8 | Candidates | L307 | (h,w) bool | forward ∧ backward; empty → empty prediction |
| 9 | Clustering | L174–176, `utils/clustering.py` L8 | (P,C) → labels (h,w), K | agglomerative on **original** normalised target features, distance threshold 1 − τ |
| 10 | Cluster prototypes | L178–181 | (K,C) | normalised mean of **debiased** tokens per cluster |
| 11 | Area weights | L328–334 | (K,) | fraction of each cluster inside the candidate mask (0 for non-candidates) |
| 12 | Seed selection | L337–340 | scalar | `argmax_k ⟨p̃_t,k, p̃⟩` over candidate clusters (paper Eq. 10–11) |
| 13 | Intra similarity | L343–347 | (K,) | `⟨p̄_t,seed, p̄_t,k⟩` with **original**-space cluster prototypes (Eq. 12) |
| 14 | Cross score | L350–354 | (K,) | **mean of token similarities** `⟨x̃_i, p̃⟩` inside each cluster |
| 15 | Combination | L357–359 | (K,) | `cross · intra · area_weight`, seed area weight forced to 1 |
| 16 | Final mask | L361–363 | (h,w) | clusters with combined **>** `merge_threshold` (0.2) |
| 17 | Upsampling | `_finalize_mask` L368–383 | (H,W) | bilinear (> 0.5) or CRF; evaluation uses `resize_to_orig_size=False` → `image_size` |

Keypoint transfer (`predict_keypoint` L194–228): normalised features, optionally debiased, descriptor at
the rounded token of each source keypoint, cosine map over target tokens, Gaussian-suppressed soft-argmax
(temperature 0.04, σ = 7 tokens), rescaled to the original target size. PCK: per-image, threshold
`α · max(bbox_w, bbox_h)` of the target box.

```
 reference image(s) + mask(s)                 target image
          │                                         │
          └──────────────► DINOv3 (frozen) ◄────────┘          (1) features, (2) L2
                               │
              ┌────────────────┴────────────────┐
        original features                 debiased features  (3)  x̃ = normalize((I−BBᵀ)x)
              │                                  │                   └─ TASK 1 replaces this
              │                     ┌────────────┼──────────────────────────┐
              │              ref prototype p̃ (4)   dense ref↔target sims (5)  cluster protos p̃_t,k (10)
              │              └─ TASK 2 replaces │                           │
              │                 forward mask (6) backward NN votes (7)      │
              │                          └── candidates (8) ──┐             │
   agglomerative clustering (9) ──────────────────────────────┼─────────────┤
              │                                               │             │
   intra-image sim to seed (13)        seed = argmax ⟨p̃_t,k, p̃⟩ (12) ◄──────┘   ← TASK 2
              │                        cross score mean⟨x̃_i, p̃⟩ (14)          ← TASK 2
              └──────────────► combined = cross·intra·area (15) > α (16) → upsample (17)
```

## 3. Exact locations of the two modifications

* **Task 1** — the single point of change is `INSID3._debias_features` (L263) plus the basis
  construction `_build_positional_basis` (L246). Every downstream use of debiased features (steps 4, 5,
  6, 7, 10, 12, 14 and keypoint transfer) inherits the change. Implemented by overriding these two
  methods in `insid3_ext/model.py` — the original code path is kept for `debias.mode=hard`.
* **Task 2** — the single prototype `ref_prototype` (L150–158) is used in exactly three places:
  forward mask (step 6), seed selection (step 12) and cross score (step 14). `INSID3Ext.predict_mask`
  replaces these three uses; backward NN, clustering, intra similarity, area weights and thresholding are
  copied verbatim (`_backward_mask`, `_seed_and_aggregate_scores`).

## 4. Differences between paper text and code (code is reproduced; paper noted)

| # | Topic | Paper | Code (what we reproduce) |
|---|---|---|---|
| D1 | Image for the positional SVD | "noise image I ~ N(0,1)" (Eq. 3) | `torch.zeros` image passed through ImageNet `normalize` = constant image (L249–252). Supp. A.2 reports nearly identical results for other low-content images. |
| D2 | SVD factor | right singular vectors of F_noise (P×D) | left singular vectors of the centred Eᵀ (C×P) — mathematically equivalent; **code centres** the tokens, the paper does not mention centring |
| D3 | Forward filter | not in Eq. 7 | candidates also require `⟨x̃, p̃⟩ > 0` (quantile fallback) |
| D4 | Cross score in aggregation | Eq. 10 prototype similarity | Seed uses Eq. 10; aggregation uses the mean of **token** similarities in the cluster (L350–354) |
| D5 | Area weighting | not in Eq. 13–14 | combined score multiplied by the fraction of the cluster inside the candidate mask |
| D6 | Threshold | `S_k ≥ α` | strict `>`; seed is also thresholded (enables empty predictions, Supp. Tab. 10) |
| D7 | Mask refinement | Tab. 1 uses CRF | scripts default to bilinear; CRF via `--crf-mask-refinement` |
| D8 | Evaluation resolution | — | prediction and GT compared at `image_size × image_size` (GT nearest-resized) because `build_insid3_from_args` sets `resize_to_orig_size=False` |
| D9 | Correspondence setting | Tab. 2: S/B/L, s not stated in the parts read | README: ViT-B, 768 px, s = 20; script default s = 500 |
| D10 | Precision | Supp. D: runtimes "in 32-bit precision" | encoder runs under bf16 autocast on CUDA; positional basis in fp32 |

## 5. Reproducibility-relevant behaviour found during the audit

1. **Episode sampling uses the global numpy RNG** (`np.random.choice` in every dataset's `__getitem__`).
   The sequence is fixed by `seed=0` *only if nothing else draws from numpy* and `num_workers=0`.
   Consequence for this project: our modules never touch the global RNGs (k-means uses a local
   `torch.Generator`; tested), and `data.num_workers` is forced to 0 by config validation.
2. `compute_cluster_prototypes` fills empty clusters with `torch.randn` (global torch RNG). Agglomerative
   labels are contiguous, so this branch is unreachable in practice.
3. `downsample_mask` crashes on a completely empty mask (mean of an empty `argwhere`). The benchmarks
   never provide empty reference masks; we keep the official behaviour.
4. PerMIS episode order follows `os.listdir` (filesystem dependent). mIoU is order-invariant; paired
   comparisons are only valid between runs on the same machine/filesystem (fingerprints check this).
5. The official scripts append a timestamp to the output directory, so runs cannot be resumed. Our runner
   uses deterministic directories and per-episode JSONL logs instead (resume + fingerprint check).
6. `agglomerative_clustering` calls `.numpy()`, which rejects bf16 tensors (tested). Since the official
   pipeline runs on CUDA under bf16 autocast, the features it receives must be fp32 in practice — most
   likely because DINOv3's final LayerNorm is executed in fp32 by autocast. *Inference, not verified on GPU*;
   our modules accumulate in fp32 anyway (`tests/test_dtype_bf16.py`).
7. `datasets/coco.py` reads `splits/val/fold{f}.pkl`; image names include their sub-folder. We assume the
   val split lists `val2014` images (HSNet convention); the tuning split (`splits/trn`) is checked for
   overlap at run time and the count is written to `log.txt`.

## 6. Published reference numbers (from the paper/README — not produced by this project)

| Setting | Value | Source |
|---|---|---|
| COCO-20ⁱ 1-shot mIoU (CRF) | 57.6 | Tab. 1 / Tab. 3 |
| PASCAL-Part 1-shot mIoU | 50.5 | Tab. 1 / Tab. 3 |
| Average over 9 benchmarks | 55.1 | Tab. 1 |
| COCO-20ⁱ, no debiasing | 54.5 | Supp. Tab. 5 |
| SPair-71k ViT-B 768 px s=20, original / debiased PCK@0.10 | 46.8 / 52.6 | README |
| Runtime per episode, RTX 4090, 1024 px, incl. CRF | 302 ms | Supp. Tab. 12 |

These are the targets for the baseline-reproduction check (EXPERIMENTAL_PROTOCOL.md §6).
