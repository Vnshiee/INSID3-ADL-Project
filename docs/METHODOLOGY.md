# Methodology

Formulas are in MATHEMATICAL_FORMULATION.md; code locations in IMPLEMENTATION_GUIDE.md.
Each module is described with the nine points you need to defend it: intuition, formulation, tensor
view, implementation, why chosen, alternatives, limitations, expected behaviour, server validation.

**Status labels used in this project:** *Original* (official INSID3), *Proposed* (implemented extension),
*Hypothesis* (expected prior to execution), *Validated / Observed* (added when the results are compiled).

---

## Module A — Positional component analysis (Task 1, positional statistics)

1. **Intuition.** Before softening the projection we must know *how positional* each removed direction
   really is in natural images, and whether this varies across images and tokens.
2. **Formulation.** Additive model c_k(I,p) = π_k(p) + e_k(I,p); estimate template π̂_k(p), positional
   energy v̂_k (bias-corrected) and positional share ρ̂_k = v̂_k / m_k (MATH §2.4).
3. **Tensor view.** U (C×R) from the constant image; features (N,C,h,w) → coefficients (N,R,P); running
   float64 sums (R,P); per-image second moments (N,R); per-token energy in the top-s subspace (P,).
4. **Implementation.** `PositionalStatsAccumulator` (streaming), `scripts/calibrate_positional.py`
   (500 COCO train2014 images by default, resized exactly like inference, bf16 autocast like inference),
   plots via `scripts/visualize.py calibration`.
5. **Why.** It turns "the positional subspace" from a fixed rank-500 set into a measured per-component
   quantity, and its outputs (ρ̂, π̂) directly parametrise modes `calibrated`, `image`, `token`.
6. **Alternatives.** PCA of natural-image positional means (a *new* basis rather than INSID3's);
   regressing features on Fourier position features; estimating ρ by probing similarity maps.
   We kept INSID3's basis so that the analysis is about *the components INSID3 identified* (topic.md).
7. **Limitations.** Assumes position/content are uncorrelated (false for e.g. sky at the top of images —
   such correlations are *counted as positional*); calibration images are COCO-like; estimates depend on
   resolution and backbone (a file is valid for one (model_size, image_size) pair, enforced at load).
8. **Expected (hypothesis).** ρ̂ high for the leading components and decaying with k; per-token
   positional energy higher in low-texture regions/borders; images vary in how strongly they express
   the leading templates.
9. **Validate.** Inspect `calibration_spectra.png` (ρ vs k, energies), `calibration_tokens.png`,
   `calibration_templates.png` (templates should look like smooth coordinate patterns); summary JSON:
   share of positional energy captured in the top-500 components.

## Module B — Adaptive positional debiasing (Task 1)

1. **Intuition.** Hard projection removes everything in 500 directions, including semantics that
   overlap them. Remove *as much as is positional*, not everything.
2. **Formulation.** x̃ = normalize(x − U(λ⊙c)), with λ from: fixed α (weighted), spectral Wiener schedule,
   MMSE ρ (calibrated), per-image Wiener (image), per-token template subtraction (token).
3. **Tensor view.** (B,T,C,h,w) → (B·T,C,P); c = einsum('cr,ncp→nrp'); λ is (r,), (N,r) or a (N,P)
   gain times an (r,P) template; reconstruction einsum('cr,nrp→ncp'); re-normalise over C.
4. **Implementation.** `PositionalDebiaser` (one class, mode switch). `INSID3Ext._debias_features` calls
   the official method for `hard` and the debiaser otherwise. Same dtype as the input (the official code
   debiases in the feature dtype); statistics in fp32.
5. **Why.** One family covers all four strategies in topic.md, contains the original as a special case
   (λ = 1, or γ → ∞), and every member has an explicit optimality argument under a stated assumption.
   Recommended primary: `calibrated` (MMSE, no rank hyper-parameter, β = 1 needs no tuning);
   secondary: `token` (tests the strongest, position-deterministic assumption).
6. **Alternatives.** Learned projections (violates training-free); test-time augmentation averaging
   (paper Supp. A.2: needs several forward passes); whitening of the full feature covariance (mixes
   semantic directions); per-dataset tuning of s (violates the fixed-hyper-parameter protocol).
7. **Limitations.** All gains are linear per component; the downstream pipeline is thresholded and
   non-linear, so MSE optimality in feature space does not imply mIoU gains. `image` mode with noise
   variance may over-suppress; `token` assumes stable templates across images.
8. **Expected (hypothesis H1).** Small mIoU gains on part-level and correspondence-heavy benchmarks
   where semantic detail matters; PCK on SPair-71k ≥ hard projection; risk of losing some of the
   hard projection's gain if too little position is removed (β < 1).
9. **Validate.** SPair-71k PCK per mode (`STAGE=corr ./run_task1.sh`) — the most direct measure of
   cross-image matching; COCO tuning sweep; then the selected mode on all benchmarks with paired bootstrap.

## Module C — Multi-prototype reference (Task 2)

1. **Intuition.** A reference region often has several appearances (parts, textures). One mean vector
   blurs them; several prototypes let each target part find its own match.
2. **Formulation.** Spherical k-means (or agglomerative) on reference foreground tokens; normalised
   debiased means as prototypes, masses as weights (MATH §3.2).
3. **Tensor view.** Foreground tokens (N,C) per shot → labels (N,) → prototypes (M,C), weights (M,),
   shot index (M,), plus the original global prototype (C,).
4. **Implementation.** `build_reference_prototypes`, `spherical_kmeans`. Grouping in the original space,
   representation in the debiased space. Local RNG only.
5. **Why.** Spherical k-means is the clustering that matches the cosine similarity used for matching
   (it maximises within-cluster cosine); agglomerative variant mirrors INSID3's own clustering and gives
   an adaptive count.
6. **Alternatives.** GMM / von Mises–Fisher mixtures (soft assignments, more parameters); spectral
   clustering; all foreground tokens as "prototypes" (= backward NN already used by INSID3; costly and noisy).
7. **Limitations.** k is a hyper-parameter; small masks give unreliable clusters (handled by collapsing to
   one prototype); clusters may split by *position* inside the reference if grouping features keep position.
8. **Expected (hypothesis H2).** Gains for large, heterogeneous references (full objects, COCO/LVIS) and
   little effect on small part masks (often M_eff = 1).
9. **Validate.** M sweep on tuning episodes; per-GT-area analysis (`scripts/analyze_failures.py`);
   `scripts/visualize.py prototypes` for qualitative checks.

## Module D — Prototype aggregation and many-to-many matching (Task 2)

1. **Intuition.** Given several prototypes, decide how a target token / region is scored: by its best
   matching part (max), by the average (weighted), something in between (lse), or by how well a region
   and the reference *explain each other* (Chamfer, OT).
2. **Formulation.** MATH §3.3–3.4.
3. **Tensor view.** sims (P,M); token scores (P,); per-cluster reductions with `index_add_`/`scatter_reduce`
   to (K,) or (K,M); OT potentials f (P,), g (K,M).
4. **Implementation.** `aggregate_similarities`, `region_scores_chamfer`, `region_scores_ot`;
   wired in `INSID3Ext._region_scores` and the forward mask.
5. **Why.** max is the simplest non-linear rule; lse is a smooth, tunable bridge with a provable ordering;
   unbalanced OT is the standard formulation of many-to-many matching with partial coverage, and its ρ
   interpolates between "best part" and "whole reference" matching.
6. **Alternatives.** Hungarian (one-to-one) matching of target clusters to prototypes; attention-style
   softmax over target tokens; learned aggregation (not training-free).
7. **Limitations.** Larger scores interact with the fixed threshold α (MATH §3.5); balanced OT penalises
   INSID3's part-level clusters (each covers only part of the reference); OT adds ε, ρ hyper-parameters.
8. **Expected (hypothesis H3).** lse/max ≥ weighted; OT useful only with small–moderate ρ.
9. **Validate.** Tuning sweep (aggregation × M × α), then selected config on all benchmarks.

## Module E — Integration (Phase 5)

Prototypes are built from the (adaptively) debiased features so that the matching space is consistent for
prototypes, cluster prototypes and target tokens; reference tokens are *grouped* in the original space.
Backward correspondence, clustering, intra-image similarity, area weights and threshold are unchanged.
Question investigated: does Task 1 change which reference parts Task 2 finds (ablation `c_group_debiased`)
and are the two gains additive (`c_selected` vs Task-1-only and Task-2-only)? *Hypothesis H4:* partially
additive, because Task 1 changes the matching space and Task 2 the reference representation.

## Hypotheses

| ID | Hypothesis | Primary evidence |
|---|---|---|
| H1 | Adaptive debiasing ≥ hard projection | SPair PCK; COCO/PASCAL-Part mIoU, paired CI |
| H1b | ρ̂ decays with k; the hard rank-500 cut removes components with substantial semantic energy | calibration plots |
| H2 | Multi-prototype gains grow with reference size/heterogeneity | ΔIoU by GT-area tercile |
| H3 | Non-linear aggregation (max/lse) beats weighted | tuning sweep |
| H4 | Tasks 1 and 2 are (partly) additive | combined vs single-task runs |

Outcomes are added when the results are compiled. A hypothesis counts as strictly supported only if the test-set paired 95 % CI excludes 0 on the main benchmarks
and the sign is consistent across benchmarks (EXPERIMENTAL_PROTOCOL.md §7).
