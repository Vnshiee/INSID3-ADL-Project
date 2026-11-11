# 2. Our method: ideas and maths

Notation: x = one patch feature (C = 1024 numbers, unit length). ⟨a, b⟩ = dot product = cosine similarity
for unit vectors. S = number of shots. Each formula names the file and function that implements it.

---

## 2.1 Task 1 — adaptive positional debiasing  (`insid3_ext/debiasing.py`)

### One formula for all modes

Take the SVD directions u_1, u_2, … of the empty image (columns of U) and the coordinates of x along them,
c_k = ⟨u_k, x⟩. The official hard projection deletes c_1 … c_500 completely. We delete only a **fraction
λ_k** of each one:

```
x_debiased = normalize( x − Σ_k λ_k c_k u_k ) = normalize( x − U (λ ⊙ Uᵀx) ),     0 ≤ λ_k ≤ 1
```

* λ_k = 1: delete direction k; λ_k = 0: keep it.
* Hard projection: λ = 1 for k ≤ 500, else 0.
* No debiasing: λ = 0 everywhere.

Every mode below is just a different way to choose λ. We normalise again at the end because every later
step compares vectors by cosine (the official code does the same).

### The modes

| mode | λ_k | idea | needs |
|---|---|---|---|
| `weighted` | α (for k ≤ 500) | remove the same fraction of every direction | — |
| `spectral` | σ_k^{2g} / (σ_k^{2g} + σ_knee^{2g}) | a *soft* cut at the knee instead of a hard one | empty image only |
| `calibrated` | clip(β ρ_k, 0, 1) | remove exactly the measured positional share ρ_k | calibration file |
| `image` | clip(β v_k / m_k(I), 0, 1) | like `calibrated`, but with this image's own energy m_k(I) | calibration file |
| `token` | — (see below) | subtract the expected positional *pattern* at each position | calibration file |
| controls: `rho_threshold`, `random`, `pca`, `center` | | to find out *what* the hard projection really removes | see 2.3 |

**Spectral.** σ_k is the k-th singular value (how strongly direction k varies with position on the empty
image). Directions much stronger than the knee get λ ≈ 1, much weaker ones λ ≈ 0, and the knee gets exactly
0.5. g ("sharpness") sets how steep the curve is. g = 1 is the classical Wiener filter shape, and very large
g gives back the hard cut, so the official method is a limit case. In the code it is written as
`sigmoid(2g (log σ_k − log σ_knee))`, which is the same number but cannot overflow.

**Calibrated: why λ = ρ is the best fixed fraction.** Assume every coordinate is position plus content,
c = π + e, where π (position) and e (content) are unrelated. We remove λc and want to remove π as exactly
as possible:

```
error(λ) = E[(π − λc)²] = λ² E[c²] − 2λ E[πc] + E[π²],     E[πc] = E[π²]  (π and e unrelated)
d error / dλ = 0   →   λ* = E[π²] / E[c²] = ρ   (the positional share of the energy)
```

This is the Wiener / MMSE (minimum mean squared error) gain. β = 1 is that optimum; β > 1 removes more.

**Image.** The positional energy v_k is a property of the network (the same for every image). The content
energy differs per image (a plain sky against a busy street). So λ_k(I) = v_k / m_k(I) uses this image's
total energy m_k(I) = mean over its patches of c_k².

**Token.** If position adds a fixed pattern t_k(p) at every position p, the best correction is to subtract
that pattern, not to shrink the whole direction:

```
x_debiased(p) = normalize( x(p) − γ_p Σ_k t_k(p) u_k )
```

γ_p is either fixed (β), or fitted per patch by least squares, γ_p = ⟨c_p, t_p⟩ / |t_p|², capped at 2.
The fitted γ_p says "how strongly does this patch show the positional pattern".

### Measuring the positional share on real images  (`insid3_ext/calibration.py`)

We use 500 COCO train2014 images (never test images), at the same resolution as at inference. For each
direction k, image I and position p:

1. **Template** π_k(p) = mean over images of c_k(I, p), minus its mean over positions. Averaging over
   many images cancels the content and keeps what every image has at that position.
2. **Total energy** m_k = mean over images and positions of c_k².
3. **Positional energy** v_k = mean over positions of π_k(p)², **minus a correction**. An average of N
   images still contains a little content by chance, with variance (spread between images) / N, and this
   would look like position. So we subtract it (and never go below 0).
4. **Share** ρ_k = v_k / m_k.

Check: with features that contain no positional signal at all, ρ comes out ≈ 0 (`tests/test_calibration.py`).

---

## 2.2 Task 2 — multi-prototype matching  (`insid3_ext/prototypes.py`, `insid3_ext/matching.py`)

### The prototypes

The official prototype is the unit-length average of the reference patches inside the mask. For a person
(face, shirt, trousers) this average can lie between the parts and be close to none of them. So:

1. Split the masked reference patches into M groups with **spherical k-means**: k-means where "close"
   means high cosine similarity and every centre is rescaled to unit length.
   * Grouping uses the *original* features, like INSID3's own clustering of the target.
   * Small masks get fewer groups (at least `min_points_per_prototype` patches per group), so a tiny mask
     falls back to the single original prototype.
2. Each group gives one prototype q_j = unit-length average of its *debiased* features (the space in which
   reference and target are compared).
3. Each prototype gets a weight w_j = its share of the patches; every shot counts 1/S in total, like the
   original average of shot averages. The weights sum to 1.
4. With several shots, either cluster each shot separately (`per_shot`, round 1) or all shots together with
   patch weights 1/(S·N_s) (`pooled`, as in ASGNet). With M = 1, `pooled` gives exactly the official
   prototype for any number of shots (tested).

### Scoring a target patch against M prototypes

With s_j = ⟨x, q_j⟩:

| rule | score | meaning |
|---|---|---|
| `max` | max_j s_j | "matches its best part" |
| `weighted` | Σ_j w_j s_j | "matches the average over parts" |
| `lse` | T log Σ_j w_j exp(s_j / T) | a smooth max: T → 0 gives max, T → ∞ gives weighted |

Always weighted ≤ lse ≤ max. Why the prototypes must be unit length: with plain group means, `weighted`
would be exactly the original prototype, since a weighted sum of group means is the overall mean.

### Scoring a target region

INSID3 clusters the target into regions G_1 … G_K. Our score of a region:

* `prototype` (as the official code): the region's own prototype against the set (seed choice), and the
  mean patch score inside the region (merging).
* `chamfer`: ½ [ average over the region's patches of their best prototype + weighted average over
  prototypes of their best patch ] — "every patch fits some part, every part is found".
* `ot` (optimal transport, many-to-many). Each patch sends mass 1/n to the prototypes, at cost 1 − s_ij.
  Prototype j should receive about w_j, but only softly (penalty ρ). The score is the average similarity
  along the transport plan.
  * ρ → 0: every patch just goes to its best prototypes.
  * ρ → ∞: the region must contain *every* part in proportion w.
  * Solved with Sinkhorn iterations in log form; the factor ρ/(ρ+ε) makes the column rule soft
    (Peyré & Cuturi 2019, eq. 10.9).

### Where the prototype is used (the only places we change)

The official code uses the single prototype in exactly three places:

1. the forward filter;
2. the choice of the seed region;
3. the reference score of every region when merging.

We replace only these three. Backward matching, clustering, the similarity to the seed, the area weight
and the threshold stay the official code.

### Optional fixes we tried (round 2)

* `score_calibration=reference`: rescale the region scores so that on the *reference* patches the new score
  has the same mean as the original one (no target labels used).
* lower merge threshold α (0.10, 0.15).
* area-adaptive number of prototypes (one per 100 patches, at most 5), as in ASGNet.

---

## 2.3 The control experiments (why they exist)

The calibration showed that only about 15 % of the feature energy is positional, while the hard projection
removes 58 %. So we asked what the hard projection really removes, and built one control per possible
explanation:

| control | removes | if it works as well as hard, then … |
|---|---|---|
| `hard_s5 … s50` | only the first 5–50 directions | the effect lives in the first components |
| `rho_threshold` | the directions with ρ ≥ 0.3 | the clearly positional directions are enough |
| `random` | 500 random directions | removing *anything* of that size helps |
| `pca` | the 500 (or 20) strongest directions of real features | it is about dominant directions, not position |
| `center` | the mean feature of real images | it is about a shared offset |
| `basis_eval_mode` | hard, but the basis computed in eval mode | the training-mode randomness does not matter |

Results: `docs/5_results.md` §5.2 (c).
