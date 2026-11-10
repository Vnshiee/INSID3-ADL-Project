# Reading notes — four passes

Fifteen papers were read for this project (PDFs from arXiv; INSID3 main text and supplementary in full,
the others by the sections that matter here). Each pass ends with **what it means for this project**.


---

## Pass 1 — The baseline and its backbone

### INSID3 (Cuttano et al., CVPR 2026) — section map

| Section | Content | Where in the code |
|---|---|---|
| 3 (task definition) | frozen DINOv3, patch features F_r, F_t ∈ ℝ^{P×D} | `_extract_features` |
| 3.1 Unlocking the feature space | single prototype p_r = mean of masked reference tokens (Eq. 2); constant/noise-image SVD and orthogonal-complement projection (Eq. 3–4); "debiased features for matching, original for grouping" | `_build_positional_basis`, `_debias_features` |
| 3.2 Fine-grained clustering | agglomerative clustering of the *target*, threshold τ = 0.6 | `utils/clustering.py` |
| 3.3 Seed-cluster selection | backward nearest neighbours (Eq. 6–8), debiased cluster prototypes, seed = argmax cross-similarity (Eq. 9–11) | `_locate_candidates`, `_seed_and_aggregate` |
| 3.4 Cluster aggregation | score = cross × intra (Eq. 12–14), threshold α = 0.2 | `_seed_and_aggregate` |
| 4 Experiments | ViT-L/16, 1024 px, CRF refinement; **mIoU only** (no FB-IoU in the paper) | `inference_segmentation.py` |
| Supp. A.2 | 1 noise image ≈ 5 or 10 low-content images (57.6 vs 57.7) | — |
| Supp. A.4 | position can *help* some tasks (sky at the top) | relevant to our PASCAL fold-1 finding |
| Supp. C.1 | k-shot: majority vote over shots, average of per-shot prototypes | multi-shot code |

### Vision Transformers Need Registers (Darcet et al., ICLR 2024)

* Large ViTs (≥ ViT-L) develop **high-norm outlier tokens** (~10× norm, ~2 % of tokens) in redundant,
  low-information background patches. They store *global* information used by the model internally.
* Linear probes show these outliers hold **less** positional information than normal tokens
  (position top-1 41.7 % for normal vs 22.8 % for outliers). It is the *normal* tokens whose position is
  linearly decodable.
* Fix: a few extra "register" tokens absorb this role; artifacts disappear.

### DINOv2 (Oquab et al., TMLR 2024) and DINOv3 (Siméoni et al., 2025)

* DINOv2: patch 14, learned absolute position embeddings; the `_reg` variants add registers.
* **DINOv3 (our backbone)**: patch 16; **4 registers already built in**; **RoPE** (relative position inside
  attention, coordinates in a [−1, 1] box with random box jittering) instead of absolute embeddings;
  **Gram anchoring** keeps patch-to-patch similarity structure clean during long training.
* DINOv3 App. A.2: a different outlier type exists — a few **feature dimensions with large values that
  are constant across patches and images**. The final LayerNorm scales them down; batch norm or PCA also
  remove them; dropping them at inference does not hurt.
* Verified in the DINOv3 code (`dinov3/models/vision_transformer.py`): `get_intermediate_layers(...,
  norm=True)` is the default, so INSID3 receives the **final-LayerNorm** features that the DINOv3 authors
  recommend (the constant outlier dimensions are already scaled down, not removed). LayerNorm runs in
  fp32 under autocast, consistent with the official clustering code receiving fp32 features.

**What it means for us**
1. The positional bias INSID3 removes is **not** the register artifact problem — DINOv3 has registers.
   It is an absolute-coordinate signal that emerges in normal tokens even though the model only sees
   *relative* positions (RoPE); INSID3 attributes it to Gram anchoring (Sec. 3.1). Frame Task 1 that way.
2. A component shared by all tokens inflates every cosine similarity equally. This motivates a cheap
   control we added: **mean-centring** the features before matching (`debias.center=true`).

---

## Pass 2 — Task 1: position in ViT features, correspondence, Wiener/MMSE

### Deep ViT Features as Dense Visual Descriptors (Amir et al., ECCV 2022)

* In DINO-ViT, **shallow layers are mostly positional**, deep layers mostly semantic; keys/queries carry
  more positional bias than values/tokens in intermediate layers.
* For point correspondence they **want some positional bias** (mid-layer features), to separate points that
  are semantically identical (e.g. along a tail); plus log-binning for context and "best buddies" pairs.

### SPair-71k (Min et al., 2019)

* 18 PASCAL categories, 12,234 test pairs, annotated viewpoint/scale/truncation/occlusion difficulty.
* PCK@α_bbox: a predicted keypoint is correct if within α · max(h, w) of the *target* bounding box
  (exactly `pck_thresh` in `datasets/spair.py`). INSID3 averages PCK **per image pair** (image-wise PCK).

### Wiener / MMSE (classical estimation theory)

* For x = s + n with uncorrelated s, n, the best linear estimate of s is (σ_s² / (σ_s² + σ_n²)) · x.
* Our `calibrated` mode applies this with s = positional part: λ_k = σ²_pos,k / (σ²_pos,k + σ²_sem,k) = ρ_k,
  per component. `image` mode does it per component **per image**; `token` mode subtracts the expected
  positional template **per location**. The fraction *kept* is 1 − λ (the "w" in the reading list).
* Assumptions that matter: additive, uncorrelated, and — for `image`/`token` — variation that is worth
  adapting to.

**What it means for us** (together with our calibration figures)
1. Our calibration shows the leading components (ρ ≈ 0.9 for k = 1–3) are smooth coordinate patterns
   (vertical gradient, centre-vs-border, horizontal gradient) and are **the same in every image**
   (template correlation ≈ 0.95). Stationary bias ⇒ nothing to adapt per image ⇒ explains why `image`
   mode never helped. This is our finding, not Amir et al.'s; cite our figures and INSID3 Fig. 4c.
2. Amir et al. explain a pattern we saw on SPair: removing *less* position (`token_ls`) helps the strict
   PCK@0.05 (fine localisation) but hurts PCK@0.20. Position is useful for disambiguation; the right amount
   is task-dependent (also INSID3 Supp. A.4, and our PASCAL-Part fold-1 result).
3. Most energy INSID3 removes is not positional by our measure (~77 %). The controls `random`, `pca`,
   `rho_threshold`, `center` test what the projection really removes.

---

## Pass 3 — Task 2: prototypes and optimal transport

### Prototypical Networks (Snell et al., NeurIPS 2017)

* One prototype per class = the mean of its support embeddings. For Bregman divergences (e.g. squared
  Euclidean) the mean is the optimal single representative, and the method equals mixture density
  estimation with **one component per class**. A single prototype is the right model only if the class is
  unimodal in feature space. (On the unit sphere, the normalised mean maximises total cosine — this is
  exactly INSID3's prototype.)

### ASGNet — Adaptive Prototype Learning and Allocation (Li et al., CVPR 2021)

* Multiple prototypes by superpixel-guided clustering (SLIC-style: feature distance + spatial distance),
  allocation by per-pixel argmax prototype (≈ our `max`).
* **Prototype count adapts to mask area**: N = min(⌊N_mask / 100⌋, N_max), N_max = 5.
* **Key ablation**: in **1-shot**, multiple prototypes *hurt* (58.96 → 58.31 mIoU) — "prototypes from one
  support image become highly similar and cosine distance cannot distinguish them"; the gain appears in
  **5-shot** (60.19 → 61.24, 63.94 with their trained allocation). N_max 1 → 5 adds only +0.55 even with
  training.
* For k-shot they cluster all shots' features together.

### Computational Optimal Transport (Peyré & Cuturi, 2019), Sinkhorn Distances (Cuturi, 2013), Chizat et al. (2018)

* Entropic OT: min ⟨C, P⟩ − εH(P); Sinkhorn's matrix scaling: **§4.1–4.2**.
* Unbalanced OT: marginals penalised by KL with weight τ instead of hard constraints: **§10.2**; the scaling
  update becomes u ← (a ⊘ Kv)^{τ/(τ+ε)} — identical to the `ρ/(ρ+ε)` exponent in `insid3_ext/matching.py`
  (verified against Eq. 10.9).

### SuperGlue (Sarlin et al., CVPR 2020)

* Sinkhorn on a score matrix augmented with a **dustbin** row/column so unmatched points can go "nowhere".
  Our OT keeps every target token assigned and relaxes the *prototype* marginal instead; a dustbin for
  background tokens is a possible extension (not implemented).

**What it means for us**
1. Our negative 1-shot Task-2 result **agrees with ASGNet**. Report it as consistent with the literature,
   not as a failure of implementation.
2. Two literature-backed variants are now in the revision plan: area-adaptive prototype count
   (`min_points_per_prototype=100`, `num_prototypes=5`) and pooled multi-shot clustering for 5-shot.

---

## Pass 4 — Protocol and context

### OSLSM (Shaban et al., BMVC 2017) and PFENet (Tian et al., TPAMI 2020)

* PASCAL-5ⁱ: 20 classes → 4 folds of 5; 1,000 test episodes (OSLSM).
* COCO-20ⁱ: 80 classes → 4 folds of **20** classes (60 seen / 20 unseen); fold i = classes {4x − 3 + i}.
  INSID3's `datasets/coco.py` uses class ids `fold + 4v` (same interleaved split, 0-based).
* **PFENet Tab. 10: with 1,000 episodes per fold, COCO fold-0 mIoU varies with std ≈ 1.34 across sampling
  seeds; with 20,000 episodes std ≈ 0.16.** Our effects are ≤ 0.6 mIoU — below the seed noise of the
  1,000-episode protocol. Hence: keep the official 1,000 episodes for comparability, and add seeds 1–2 as
  secondary evidence (done in `STAGE=test`), plus 4,000 tuning episodes.

### Painter (Wang et al., CVPR 2023) and SegGPT (Wang et al., ICCV 2023)

* Generalist in-context models *trained* with masked image modelling on mixed segmentation data; strong
  in-domain (SegGPT COCO 56.1), weak out of domain (ISIC 37.5, PACO 13.5 in INSID3 Tab. 1).
  INSID3 (training-free, 55.1 average) beats them on average mainly through cross-domain robustness.

### Dense CRF (Krähenbühl & Koltun, NeurIPS 2011)

* Fully connected CRF with Gaussian appearance + smoothness kernels, mean-field inference in linear time.
  INSID3 cites it, but the code uses the GPU **Frank-Wolfe** CRF of Lê-Huu & Alahari (55 ms per image).
  It refines boundaries only; it does not change which object is selected.

---

## Questions the revision runs answer, linked to the reading

| Question | Reading | Stage / output |
|---|---|---|
| Is the effect of INSID3's projection in the first few components only? | INSID3 Fig. 7, our calibration | `t1` (`t1_hard_s5…50`, `t1_rho_t0.3`), `bias` |
| Is it about position at all, or about shared / dominant directions? | DINOv3 App. A.2 | `t1` (`t1_random_s500`, `t1_pca_*`, `t1_center_*`), `bias` |
| Does some position help correspondence at fine thresholds? | Amir et al. | `corr` at PCK@0.05 vs @0.20 |
| Do multiple prototypes help in 5-shot with pooled clustering and an area-adaptive count? | ASGNet | `t2` 5-shot runs, `attribution` |
| Are differences larger than sampling noise? | PFENet Tab. 10 | phase 2 `seeds`, `comparisons.md` |
