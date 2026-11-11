# 1. Background: the task, DINOv3 and INSID3

## 1.1 The task: in-context (few-shot) segmentation

We get one or a few **reference** images with a mask of an object (or object part), and a new **target**
image. We must segment "the same kind of thing" in the target. One reference = 1-shot, five = 5-shot.
One (references, target) set is called an **episode**.

The class is not known in advance. It can be a whole object (a dog), a part (a dog's leg), or something
from another domain (a skin lesion, a lung in an X-ray). So a fixed classifier is not possible.

**How it is measured.** Classes are split into **folds** (COCO-20ⁱ: 80 classes in 4 folds of 20). Each fold
has 1,000 random episodes, drawn with numpy's global random generator and seed 0. The metric is the
class-wise **mIoU**: for each class, total intersection / total union over all its episodes, then the
mean over classes, then the mean over folds.

**Semantic correspondence** (SPair-71k) is a second task with the same features. Find, in a target image,
the point that corresponds to a keypoint in a source image (e.g. "left eye"). The metric is PCK@α: a point
is correct if it lies within α · (size of the object box) of the true point.

## 1.2 DINOv3 features

DINOv3 is a Vision Transformer trained without labels. It cuts the image into 16 × 16 patches and gives
every patch a vector of C numbers (C = 1024 for ViT-L). At 1024 × 1024 px that is a 64 × 64 grid
(P = 4,096 patches).

The useful property: patches showing the same kind of thing have **similar vectors**, even in different
images (high cosine similarity). INSID3 builds a whole segmentation method on this, with no training.

The problem: the vectors also contain **where** the patch is (e.g. "near the top border"). Two unrelated
patches at the same position in two images then look similar. The transformer gets position through RoPE
(rotary position embedding), which only encodes relative position. The final features still contain a
smooth absolute-position signal, which the INSID3 paper links to DINOv3's training.

## 1.3 INSID3 (Cuttano et al., CVPR 2026), step by step

The paper's main rule: **compare across images with debiased features; work inside one image with the
original features** (inside one image, position is useful because neighbouring patches belong together).

1. **Positional debiasing (Sec. 3.1).**
   * Put an empty image (all zeros after the usual ImageNet normalisation) through DINOv3. Its patches can
     only differ by their position.
   * Normalise the patch vectors, subtract their mean, and take the SVD. The first s = 500 directions are
     the "positional subspace" B.
   * Every feature x is then replaced by `normalize((I − B Bᵀ) x)`: everything along those 500 directions
     is deleted.
2. **Reference prototype (Eq. 2).** The average of the debiased reference features inside the mask, made
   unit length. With several shots: the average of the per-shot averages.
3. **Candidate patches (Sec. 3.3).**
   * *Forward:* the target patch must have positive similarity to the prototype.
   * *Backward:* the target patch's most similar reference patch must lie inside the reference mask
     (majority vote over shots).
4. **Fine-grained clustering (Sec. 3.2).** The target is split into small coherent regions by agglomerative
   clustering of the *original* features (cosine distance, threshold τ = 0.6).
5. **Seed and merging (Sec. 3.3–3.4).**
   * The seed is the region (containing candidates) most similar to the prototype.
   * Every region gets a score = (similarity to the reference) × (similarity to the seed, original
     features) × (share of candidate patches). Regions with a score above α = 0.2 form the mask.
6. **Upsampling** of the 64 × 64 mask to the image size (bilinear, or CRF in the paper's tables).

Published results: COCO-20ⁱ 57.6, PASCAL-Part 50.5, mean over 9 benchmarks 55.1 (1-shot, with CRF).

## 1.4 Paper vs official code

We reproduce the **code**. Where it differs from the paper, we write it down rather than change it:

| # | Topic | Paper | Code |
|---|---|---|---|
| D1 | image for the SVD | noise image | constant image (zeros after normalisation) |
| D2 | SVD | of the features | of the *centred* features (mean subtracted) |
| D3 | forward filter | not in the equations | candidates also need ⟨x, p⟩ > 0 |
| D4 | reference score of a region | prototype similarity | mean of the patch similarities in the region |
| D5 | area weight | not mentioned | score × share of candidate patches |
| D6 | threshold | S ≥ α | S > α; the seed is thresholded too |
| D7 | upsampling | CRF in Tab. 1 | bilinear by default |
| D8 | evaluation size | — | 1024 × 1024 (ground truth resized) |
| D9 | correspondence setting | — | README: ViT-B, 768 px, s = 20 |
| D10 | precision | 32-bit | encoder in bf16 on the GPU, basis in fp32 |
| D11 | (found by us) basis | — | computed while DINOv3 is still in training mode, where RoPE rescales coordinates randomly; reproducible only through the seed |

## 1.5 The two tasks of this project (topic.md)

* **Task 1 — Adaptive Positional Debiasing.** The hard projection deletes all 500 directions completely,
  although some of them may also carry meaning. Suppress each positional component *according to its
  measured contribution* instead (weighted, component-wise, per image), and compare segmentation mIoU and
  correspondence quality.
* **Task 2 — Multi-Prototype Reference Matching.** One average vector may not describe a reference with
  several parts or textures. Cluster the reference patches into several prototypes, compare target regions
  with all of them (max, weighted, many-to-many), study the number of prototypes, and compare with the
  original on the same benchmarks.
