# Mathematical formulation

Notation: C feature channels, P = h·w tokens, x ∈ ℝ^C an L2-normalised DINOv3 token, S shots.
"Original" = official INSID3 code (REPOSITORY_AUDIT.md). "Proposed" = this project. Every proposed
quantity is implemented in the file named in brackets.

---

## 1. Original positional debiasing

**Basis.** Let F₀ ∈ ℝ^{C×P} be the L2-normalised tokens of the constant (zeros-after-normalisation) image,
E = F₀ − F̄₀ 1ᵀ its centred version and E = U Σ Vᵀ its SVD (U ∈ ℝ^{C×R}, R = min(C, P)).
The positional basis is B = U_{:, 1:s}, s = 500.

**Projection.** x̃ = (I − BBᵀ)x / ‖(I − BBᵀ)x‖.

Properties of P⊥ = I − BBᵀ: symmetric, idempotent (P⊥² = P⊥), eigenvalues 1 (multiplicity C − s) and
0 (s). After debiasing the cosine similarity becomes

  cos̃(x, y) = xᵀ P⊥ y / (‖P⊥x‖ ‖P⊥y‖),

i.e. every component in span(B) is ignored entirely, **including its semantic part**.

**The limitation (why Task 1 exists).** Write the coefficient of x on u_k as
c_k = u_kᵀx = π_k(p) + e_k, where π_k(p) is the positional signal at token position p and e_k is
image content. The hard projection sets the whole c_k to 0: it removes π_k (good) and e_k (bad).
Because s = 500 out of C = 1024 dimensions are removed, a large share of the feature space is
discarded. Paper Fig. 7 shows mIoU first rising and then falling with s — consistent with the tail
components carrying semantics.

---

## 2. Task 1 — Adaptive positional debiasing  [`insid3_ext/debiasing.py`]

### 2.1 General family

With c = U_rᵀ x ∈ ℝ^r (U_r = first r columns of U):

  x̃ = normalize( x − U_r (λ ⊙ c) ),   λ_k ∈ [0, 1] = fraction of component k removed.

For λ independent of the token (static modes) this is a fixed linear map W = I − U_r diag(λ) U_rᵀ with
eigenvalues 1 − λ_k ∈ [0, 1] (no amplification, so it cannot create spurious directions) and

  cos̃(x, y) ∝ xᵀ W² y,

a positive semi-definite re-weighting of the metric. Hard projection = λ = 1_s; no debiasing = λ = 0.

### 2.2 Fixed weighted projection — `mode: weighted`

λ_k = α for k ≤ s. W = I − α BBᵀ. Interpolates linearly between no debiasing (α = 0) and the original
(α = 1). One global knob; tests whether *partial* removal alone helps.

### 2.3 Data-free component-wise attenuation — `mode: spectral`

Assumption: positional energy along u_k is ∝ σ_k² (the noise-image spectrum) and semantic energy is
isotropic with level σ_⋆². The Wiener gain (2.4) is then

  λ_k = σ_k^{2γ} / (σ_k^{2γ} + σ_⋆^{2γ}) = sigmoid(2γ (log σ_k − log σ_⋆)),  σ_⋆ := σ_{knee}.

γ = 1 is the Wiener case; λ = ½ at k = knee; λ decreases monotonically with k.
**Limit:** for γ → ∞ (distinct singular values) λ_k → 1 for k < knee and → 0 for k > knee — the hard
rank-`knee` projection. So the original method is a limiting case (tested in `tests/test_debiasing.py`).

### 2.4 Component-wise MMSE attenuation — `mode: calibrated` (recommended primary method)

**Model.** For calibration images I and positions p: c_k(I, p) = π_k(p) + e_k(I, p), with
E_p[π_k] = 0 and E[π_k e_k] = 0 (position and content uncorrelated).

**Optimal gain.** We remove λ_k c_k and want the removed part to equal the positional part:

  λ_k* = argmin_λ E[(π_k − λ c_k)²] = E[π_k c_k] / E[c_k²] = E[π_k²] / E[c_k²] =: ρ_k.

ρ_k ∈ [0, 1] is the **positional share of the second moment** of component k.

**Error comparison** (per component, v = E[π²], n = E[e²]):

| method | residual error E[(π − λc)²] | semantic energy kept |
|---|---|---|
| none (λ=0) | v | n |
| hard (λ=1) | n | 0 |
| MMSE (λ=ρ) | v·n / (v + n) ≤ min(v, n) | (1−ρ)² n |

So under the additive model the MMSE gain is never worse than either extreme, and hard projection is
optimal only for components with n = 0. Whether this transfers to mIoU is **the hypothesis H1**, not
a guarantee (the downstream pipeline is non-linear and thresholded).

**Estimation** [`insid3_ext/calibration.py`] from N calibration images:

  μ_k(p) = mean_I c_k(I,p),  μ̄_k = mean_p μ_k(p),  π̂_k(p) = μ_k(p) − μ̄_k   (template)
  m_k = mean_{I,p} c_k²   (second moment)
  w_k(p) = within-position variance (unbiased, factor N/(N−1))
  v̂_k = mean_p π̂_k(p)² − mean_p w_k(p)/N   (clamped ≥ 0)
  ρ̂_k = clamp(v̂_k / m_k, 0, 1).

The subtracted term corrects the upward bias E[(μ_k(p) − μ̄_k)²] ≈ Var_p π_k + σ²_within/N
(validated on synthetic data with no positional signal: ρ̂ ≈ 0, `tests/test_calibration.py`).

Strength β: λ_k = clamp(β ρ̂_k, 0, 1); β = 1 is the MMSE point (no tuning needed), β is swept as
an ablation. Default rank: all R components (no rank hyper-parameter).

### 2.5 Image-dependent correction — `mode: image`

Same derivation per image I: the positional energy v_k is a property of the encoder (estimated once),
but content energy differs between images (a uniform sky vs a cluttered street). With
m_k(I) = mean_p c_k(I,p)² measured on the image being processed,

  λ_k(I) = clamp(β v_k / m_k(I), 0, 1).

Images whose energy along u_k exceeds the positional level keep more of component k.
`variance_source: calibration` uses v̂_k from §2.4; `noise` uses σ_k²/P from the constant image (data-free,
but likely over-estimates the positional energy present in natural images).

### 2.6 Token-dependent correction — `mode: token`

If the positional signal is a deterministic function of position, the optimal removal is not a gain at all
but **subtraction of the expected positional coefficient at that position**:

  x̃_p = normalize( x_p − γ_p U_r π_p ),   π_p = template at position p (r-vector).

* `token_gain: fixed`: γ_p = β (β = 1: exact template subtraction, leaves e_k untouched).
* `token_gain: least_squares`: γ_p = clamp(β ⟨c_p, π_p⟩ / ‖π_p‖², 0, γ_max) — the amplitude with which
  token p actually expresses its positional template. Motivation: the paper observes that positional
  bias dominates where semantic content is weak; tokens there should express the template strongly.
  Only the component of e along the single direction π_p can be lost.

Template source: calibration (π̂, natural images) or the constant image (Uᵀ E, data-free).
On the constant image itself, full-rank fixed-gain subtraction maps every token to the same vector
(test `test_token_mode_noise_template_flattens_constant_image`).

### 2.7 Summary of assumptions (from weakest to strongest)

weighted (none) → spectral (isotropic semantics) → calibrated (additive, uncorrelated, stationary over
images) → image (additive, per-image content energy) → token (deterministic positional template).

---

## 3. Task 2 — Multi-prototype reference matching  [`insid3_ext/prototypes.py`, `matching.py`]

### 3.1 Original single prototype

p̃ = normalize( (1/S) Σ_m (1/N_m) Σ_{j∈R_m} x̃_j ). Used for (i) forward mask ⟨x̃_i, p̃⟩ > 0,
(ii) seed: argmax_k ⟨p̃_t,k, p̃⟩, (iii) cross score s_k = mean_{i∈G_k} ⟨x̃_i, p̃⟩.

**Limitation.** The mean of a multi-modal set (e.g. a person: face, shirt, trousers) can lie far from
every mode on the sphere; its norm before normalisation shrinks as the modes spread. Parts similar to
a minority mode then score low.

### 3.2 Prototype construction

For shot m with foreground tokens {x_j} (grouping features g_j, matching features x̃_j):

* **spherical k-means** (cosine): maximise Σ_j ⟨g_j, q_{a(j)}⟩ over assignments a and unit centres q;
  k-means++ seeding with distance 1 − cos, local generator, re-seeding of empty clusters,
  M_eff = max(1, min(M, ⌊N_m / n_min⌋)).
* **agglomerative** (same routine and linkage as INSID3's target clustering) with threshold `ref_tau`,
  adaptive count capped at M (largest groups kept, others re-assigned).

Prototypes in the matching space: q_{m,j} = normalize(mean_{i∈G_{m,j}} x̃_i), weights
w_{m,j} = |G_{m,j}| / (N_m S) (Σ w = 1; every shot has total mass 1/S as in the original average).

**Grouping vs matching space** (`cluster_space`). Default `original`: INSID3's own principle is to use
original features for intra-image grouping (spatial structure helps) and debiased features for cross-image
matching (position hurts). Grouping in the debiased space is an ablation.

### 3.3 Token-level aggregation (forward mask and cross score)

s_ij = ⟨x̃_i, q_j⟩.

* max: a_i = max_j s_ij
* weighted: a_i = Σ_j w_j s_ij
* lse: a_i = T log Σ_j w_j exp(s_ij / T)

**Property 1 (why normalisation matters).** With unnormalised group means μ_j,
Σ_j w_j ⟨x, μ_j⟩ = ⟨x, Σ_j w_j μ_j⟩ = ⟨x, μ⟩ — the original prototype (before normalisation).
"weighted" differs from the baseline only because each q_j is normalised (tight minority parts are no
longer down-weighted by their spread). max and lse are genuinely non-linear.

**Property 2 (ordering).** By Jensen (log is concave) weighted ≤ lse ≤ max, with lse → max as T → 0 and
lse → weighted as T → ∞. Tested in `tests/test_matching.py`.

**Property 3 (reduction).** With M = 1 every rule equals the single-prototype similarity; with
`region_matching=prototype` and 1 shot the full pipeline reproduces the original
(`tests/test_baseline_equivalence.py`).

### 3.4 Region-level matching (seed selection and aggregation)

`prototype` (mirrors the original): seed_k = agg_j ⟨p̃_t,k, q_j⟩, cross_k = mean_{i∈G_k} a_i.

`chamfer` (many-to-many, no parameters):
 r_k = ½ [ mean_{i∈G_k} max_j s_ij + Σ_j w_j max_{i∈G_k} s_ij ]
(target→reference fit + reference→target coverage).

`ot` (many-to-many, unbalanced entropic optimal transport). For cluster k, a_i = 1/|G_k| on its tokens,
b = w on prototypes, cost C_ij = 1 − s_ij:

 min_{π ≥ 0, π1 = a}  ⟨C, π⟩ + ε KL(π ‖ a⊗b) + ρ KL(πᵀ1 ‖ b),   r_k = Σ π_ij s_ij / Σ π_ij.

Log-domain Sinkhorn (Chizat et al., 2018):
 f_i = −ε log Σ_j b_j exp((g_j − C_ij)/ε)  (hard row marginal),
 g_j = −ρ/(ρ+ε) · ε log Σ_i a_i exp((f_i − C_ij)/ε)  (KL-relaxed column marginal).
All K clusters are solved jointly with scatter reductions (no Python loop).

Limits (both tested): ρ → 0 ⇒ g → 0 and π_i· = a_i softmax_j(log w_j + s_ij/ε): each token picks its
best part (soft max, like lse with T = ε); ρ → ∞ ⇒ balanced OT: the cluster must cover **every** part in
proportion w. ρ therefore controls how much a region is required to explain the *whole* reference.

### 3.5 A confound to control

max/lse/OT scores are ≥ the single-prototype score, so with the same threshold α = 0.2 more clusters
may pass (larger masks). The Task-2 sweep therefore also varies α (`run_task2.sh`), and selection is
done on tuning episodes only.

---

## 4. Integration (Phase 5)

| Quantity | Space (original INSID3) | Space (proposed, default) |
|---|---|---|
| target clustering, intra-image similarity | original | original (unchanged) |
| backward NN, dense cross similarity | debiased | adaptively debiased |
| reference prototypes | debiased (1) | adaptively debiased (M) |
| grouping of reference tokens | — | original (ablation: debiased) |
| cluster prototypes for seed | debiased | adaptively debiased |

All similarities are cosines of L2-normalised vectors: features are normalised after debiasing (as in
the original), prototypes and cluster means are re-normalised after averaging.
