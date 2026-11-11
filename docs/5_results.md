# 5. Results and what they mean

All numbers come from the machine-generated tables in `results/round1/` and `results/round2/`
(`summary.md` = all runs, `comparisons.md` = paired bootstrap against the baseline). Notation:

* **Δ** = method − baseline, in mIoU points (segmentation) or PCK points (correspondence).
* **[a, b]** = 95 % confidence interval of Δ from a paired bootstrap (resampling the same episodes for both).
* **\*** = the interval does not contain 0, i.e. the difference is larger than the sampling noise.
* *Tuning* = COCO train2014 episodes (used to choose settings); *test* = the official benchmark episodes.

How to read the two rounds:

* **Round 1** (RTX 3070) swept many settings on 1,000 tuning episodes, picked the best ones by a rule fixed
  in advance, and tested them on COCO-20ⁱ and PASCAL-Part.
* **Round 2** (a larger GPU server, for speed) re-checked the round-1 findings with control experiments,
  4,000 tuning episodes and SPair-71k at the segmentation backbone. Every round-2 comparison uses a
  baseline computed in round 2 on the same machine.

---

## 5.1 The baseline is reproduced

| Check | Paper | Ours |
|---|---|---|
| COCO-20ⁱ 1-shot test | 57.6 (with CRF) | 56.13 (bilinear) |
| PASCAL-Part 1-shot test | 50.5 (with CRF) | 49.13 (bilinear) |
| COCO-20ⁱ without debiasing | 54.5 | 53.86 |
| SPair-71k ViT-B/768, s = 20, PCK@0.10 original → debiased | 46.8 → 52.6 | 46.85 → 52.57 |
| Our Task-2 code with M = 1 vs the official code | must be equal | 57.4772 = 57.4772 |

The SPair numbers match to 0.06 points. Our segmentation numbers are about 1.4 points lower. We used
bilinear upsampling, while the paper's numbers use CRF refinement. The CRF run is still to do (phase 2), so
"reproduced" holds for the SPair setting and *probably* for segmentation.

---

## 5.2 Task 1 — adaptive positional debiasing

### (a) Positional debiasing matters (both rounds agree)

| | Δ | 95 % CI |
|---|---|---|
| no debiasing, COCO test (round 1) | −2.27\* | [−3.06, −1.45] |
| no debiasing, PASCAL-Part test (round 1) | −1.05\* | [−1.52, −0.57] |
| no debiasing, COCO tuning (round 2, 4,000 episodes) | −3.60\* | [−4.42, −2.74] |

### (b) On segmentation, no soft or adaptive mode beat the hard projection

Round 1 seemed to find a winner: spectral attenuation (knee 500, γ = 4) scored +0.44 on the tuning episodes
(p = 0.04). We then checked it in three ways, and none of them confirmed it:

| Check of "spectral k500 γ4" | Δ | 95 % CI |
|---|---|---|
| round 1, COCO test | −0.00 | [−0.27, +0.27] |
| round 1, PASCAL-Part test | −0.07 | [−0.25, +0.09] |
| round 2, COCO tuning with 4,000 new episodes | −0.08 | [−0.29, +0.14] |

**Why the first result was misleading.** It was the best of 29 Task-1 settings, all tested on the same
1,000 episodes. If you try many settings that are truly equal, the best-looking one is ahead by luck (the
"winner's curse"). 940 of the 1,000 episodes did not change at all, and 15 episodes produced the whole gain.
The spectral mode with γ = 4 is also almost the same operation as the hard cut at 500, so a large
difference was never likely.

All other modes were equal to or worse than hard (round 2, tuning):

| mode | what it removes | Δ |
|---|---|---|
| calibrated β = 1 | the measured positional share of every direction | −0.59\* |
| ρ ≥ 0.3 | only the clearly positional directions (~ the first 10–20) | −0.70\* |
| hard, s = 5 / 10 / 20 / 50 | only the first s directions | −0.87\* / −0.69\* / −0.82\* / −0.45 |
| image (round 1) | per-image amount | −0.69 … −1.51 |
| token (round 1) | the positional pattern at each position | −0.41 … −1.01 |

### (c) Where the benefit of the hard projection comes from (the control experiments)

We wanted to know *what* the 500 removed directions contain. The calibration on 500 real images showed:

* The positional signal is concentrated in very few directions: the share ρ is about 0.9, 0.7, 0.9 for
  directions 1–3, above 0.2 only for roughly the first 10–20, and a flat 0.03–0.05 after that.
* The 500 removed directions carry **58 %** of the energy of real-image features, but only **15 %** of all
  feature energy is positional. So most of what the hard projection removes is *not* positional by our
  measure.
* The positional patterns are smooth (top–bottom gradient, centre–border, left–right gradient), strongest
  at the image border, and nearly identical in every image (correlation ≈ 0.95 for the first 4 directions).
  That is why the per-image mode had nothing to adapt to.

Three explanations were possible. The controls (round 2, tuning, Δ vs hard s = 500) decide between them:

| Hypothesis | Control | Result | Verdict |
|---|---|---|---|
| "Removing *any* 500 directions helps" | 500 random directions | −3.46\* (as bad as no debiasing) | **rejected** |
| "It removes the dominant directions of real features" | top-500 / top-20 PCA directions of real images | −22.58\* / −2.19\* | **rejected** (it destroys meaning) |
| "It removes a common offset shared by all features" | subtract the mean feature | −4.67\* (worse than none); with hard: −1.28\* | **rejected** |

The bias measurement (`results/round2/positional_bias.md`, 150 pairs of unrelated images) shows the same
thing directly. The random subspace leaves the same-position bias untouched (0.149 vs 0.151 without
debiasing), and centring even increases it (0.195).

**What remains.** The specific empty-image directions matter. The first 5 recover about three quarters of
the total gain (from −3.60 to −0.87), and the remaining directions up to 500 add the last ~0.9 point,
although our measurement calls them almost non-positional.

> **Careful interpretation.** We cannot say *why* the weak directions 20–500 help segmentation. A possible
> reason is that INSID3's thresholds (τ = 0.6, α = 0.2) were chosen by the authors together with s = 500.
> Every alternative mode changes the similarity scale those thresholds act on. We did not re-tune τ and α
> per mode, so our segmentation comparison is a "plug-in" comparison.

### (d) On semantic correspondence, adaptive debiasing is clearly better

SPair-71k, 12,234 image pairs, segmentation backbone (ViT-L/16, 1024 px). Reference: hard projection s = 500
(PCK@0.10 = 54.93).

| mode | PCK@0.05 | PCK@0.10 | PCK@0.15 | PCK@0.20 |
|---|---|---|---|---|
| **token** (subtract the positional pattern) | **+2.39\*** | **+2.41\*** [+2.15, +2.68] | **+1.84\*** | **+1.36\*** |
| calibrated β = 1 | +1.92\* | +2.10\* | +1.61\* | +1.22\* |
| ρ ≥ 0.3 | +1.75\* | +1.89\* | +1.41\* | +1.05\* |
| hard s = 20 (absolute PCK) | 38.73 | 56.72 | 66.15 | 72.29 |
| spectral k500 γ4 | +0.03 | +0.01 | +0.05 | −0.04 |
| hard, basis in eval mode | +0.10 | +0.16 | −0.02 | −0.04 |
| no debiasing | +0.57\* | −0.22 | −0.98\* | −1.66\* |

*(hard s = 20 has a different s, so the script does not pair it with s = 500. Absolute values for comparison:
37.08 / 54.93 / 64.78 / 71.25.)*

* The three modes that remove **mainly the positional part** win at every threshold, with tight intervals.
* The bias measurement agrees. The token mode removes almost all same-position bias (0.005, against 0.040
  for hard) while keeping 84 % of the feature energy (hard keeps 42 %).
* In round 1 (ViT-B, 768 px, s = 20, the README setting) all modes were within ±0.6. There the hard
  projection removes only 20 directions, so there was little room to improve.

**Conclusion for Task 1.** The right amount of removal depends on the task. For correspondence, removing
exactly the positional part is best (+2.4 PCK@0.10). For INSID3's segmentation pipeline, the original hard
projection of 500 directions remains best, and its benefit goes beyond what we can measure as "position".

### (e) Two side findings

* **The official basis is computed in training mode.** The model is built before `model.eval()` is called,
  and in training mode DINOv3 rescales the patch coordinates randomly. So the official basis comes from one
  random draw (fixed only by the seed). Computing it in eval mode changes nothing on SPair (+0.16, not
  significant), but costs −0.60\* on segmentation. With one draw we cannot tell whether that draw was lucky
  or whether training-mode bases are better in general.
* **Positional information can help.** On PASCAL-Part fold 1 (indoor parts, images cropped around the object),
  *no* debiasing was better than the baseline (54.7 vs 53.2, round 1). Without debiasing, SPair at the
  strictest threshold is also better (+0.57\* at PCK@0.05). Position is informative when objects are
  centred or the match must be very precise.

---

## 5.3 Task 2 — multi-prototype reference matching

### (a) 1-shot: more prototypes never helped

Round 1, tuning (baseline 57.48):

| aggregation | M = 2 | M = 3 | M = 4 | M = 6 | M = 8 |
|---|---|---|---|---|---|
| max | 57.25 | 56.39 | 55.94 | 55.79 | 55.82 |
| weighted | 57.20 | 57.03 | 56.64 | 56.50 | 56.31 |
| log-sum-exp (T = 0.05) | 57.56 | 57.19 | 56.69 | 56.48 | 56.38 |

The many-to-many rules were not better: Chamfer gave 56.95, the best optimal transport 57.32, and agglomerative
grouping 55.90–56.71.

The best setting (log-sum-exp, M = 2, +0.08, p = 0.95) was selected and tested. Test: COCO −0.19 (not
significant), PASCAL-Part −0.62\*. Round 2 with 4,000 episodes: **−0.31\***. It is not an improvement.

### (b) Why: our first explanation was wrong

Our first idea was that more prototypes make the scores too high, so background leaks into the mask.
The data say the opposite:

* When M = 2 changed an episode, the predicted mask became **smaller** in 53 cases and larger in only 18
  (round 1). In the 5-shot tests of round 2 it was smaller 1,357 times against 346.
* Raising the threshold α made things much worse (M = 4: 54.0 / 52.2 / 49.2 for α = 0.25 / 0.30 / 0.35).
* Lowering α did not help either: α = 0.15 gives −0.35 (not significant), α = 0.10 gives −1.76\* (round 2).
  Rescaling the scores to the original scale on the reference patches also gave −0.98\*.

A possible reason, which we have not proven: the original prototype is the *re-normalised* mean of all
reference patches. Averaging different parts makes the mean short, and re-normalising scales it up, which
favours "average-looking" target regions. Each part prototype matches only its own part. So fewer regions
pass the merge rule, and the mask shrinks. Since lowering α does not repair it, the cause is not only the
threshold.

### (c) 5-shot: Task 2 caused the round-1 drop, and pooling repairs most of it

In round 1 only the combined method was tested with 5 shots (COCO −0.48\*, PASCAL-Part −1.85\*), so we did
not know which task caused the loss. Round 2 tested each task alone:

| 5-shot test | COCO Δ | PASCAL-Part Δ |
|---|---|---|
| Task 1 alone (spectral) | +0.02 [−0.17, +0.22] | +0.01 [−0.07, +0.10] |
| Task 2 alone (LSE, M = 2, one clustering per shot) | −0.66\* [−1.04, −0.22] | −1.81\* [−2.16, −1.47] |

So the loss comes entirely from Task 2. The worst fold is PASCAL-Part fold 1 (indoor), 67.0 → 62.0.

With 5 shots, clustering every shot separately gives 10 prototypes and changes how much each shot counts.
Clustering all shots together ("pooled", as in ASGNet) is close to the baseline (round 2, 5-shot tuning):

| 5-shot tuning | Δ |
|---|---|
| one clustering per shot, M = 2 | −0.82 [−1.41, +0.09] |
| pooled, M = 2 | −0.14 [−0.57, +0.35] |
| pooled, area-adaptive M | −0.43 [−1.00, +0.22] |

**Conclusion for Task 2.** With INSID3's fine part-level clustering of the target, one prototype per
reference works at least as well as several. This matches ASGNet (CVPR 2021), where multiple prototypes
also lowered 1-shot accuracy and only helped with 5 shots *and* a trained allocation module. Without
training, pooled prototypes only reach the baseline.

---

## 5.4 Reproducibility

* The same baseline run on two different GPUs gives the same mean (−0.03 mIoU) but only **34.6 %** of the
  episodes are identical. Tiny floating-point differences flip a few clustering and threshold decisions.
  This is why every comparison pairs runs from the same machine only.
* The calibration of round 2 gives almost the same numbers as round 1 (58.1 % vs 57.4 % of the energy in
  the top 500, 14.6 % positional in both, mean ρ 0.082 vs 0.083). The small difference is expected: the
  round-1 calibration script did not set the seed before building the model, so its basis came from a
  different random draw (see 5.2 e). Since round 2, every script sets the seed first.
* Every number in this file can be traced to a run directory (`config.yaml`, `env.json` with the hardware,
  `episodes.jsonl` with every episode, `metrics.json`).

---

## 5.5 Limitations (what these results do NOT show)

1. **Benchmarks:** only 2 of the 9 segmentation benchmarks (COCO-20ⁱ, PASCAL-Part) and SPair-71k so far.
   The others (LVIS, PACO-Part, iSAID, ISIC, Chest X-ray, SUIM, PerMIS) are planned (phase 2).
2. **No CRF:** all segmentation numbers use bilinear upsampling; the paper uses CRF.
3. **One episode seed:** the official protocol has 1,000 episodes per fold, and the mIoU of one fold
   varies by about ±1.3 between seeds (PFENet). Paired comparisons remove most of this, but seeds 1 and 2 are
   planned.
4. **Thresholds not re-tuned:** τ, α and s were kept at the official values for every mode (see 5.2 c).
5. **Tuning only on COCO:** all choices were made on COCO train2014; a dataset-specific choice (e.g. less
   debiasing for PASCAL-Part fold 1) was not tried.
6. **Runtime:** round-2 runs shared the GPU with other jobs, so their times are not valid. Runtime claims
   use round 1 (RTX 3070: ~546 ms per 1-shot episode, ≤ 10 ms extra for our methods, 1.5 GB GPU memory at
   1-shot and 2.3 GB at 5-shot).

---

## 5.6 Summary in five sentences

1. Positional debiasing is essential for INSID3 (−2.3 to −3.6 mIoU without it).
2. On segmentation, the original hard projection could not be beaten. The positional signal itself sits in
   the first ~20 directions, but the extra directions up to 500 still help, for reasons we could narrow down
   but not prove.
3. On semantic correspondence, removing only the positional part is clearly better (+2.4 PCK@0.10 at
   ViT-L/1024), so Task 1's idea works where matching quality is measured directly.
4. Multiple reference prototypes did not help in 1-shot and caused the 5-shot loss of round 1. Pooling the
   shots removes most of the loss but brings no gain.
5. Round 1's apparent improvements were selection noise; testing them again with more episodes and controls
   was the most important step of the project.
