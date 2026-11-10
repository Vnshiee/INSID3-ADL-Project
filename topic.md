# Research Objectives: Adaptive Debiasing and Multi-Prototype Matching in INSID3

**Base Codebase:** https://github.com/visinf/INSID3

## Overview
INSID3 [1] is a training-free in-context segmentation framework built on a frozen DINOv3 backbone. The pipeline removes positional bias from dense patch features, decomposes the target image into coherent candidate regions, and matches those regions with an annotated reference example using semantic correspondence and foreground prototypes — enabling segmentation without fine-tuning, decoders, or auxiliary models.

This project investigates two extensions to the original INSID3 pipeline:

---

## 1. Adaptive Positional Debiasing

### Motivation
INSID3 estimates a low-dimensional positional-bias subspace via SVD on blank-image features and removes it uniformly by projecting every token onto its orthogonal complement. Applying the same hard projection across all components and images may suppress valid semantic texture or boundary information that overlaps with those directions.

### Objectives
- Analyze the empirical contribution of INSID3's positional components across natural images and spatial patch positions.
- Replace the hard orthogonal-complement projection with a parameterized, adaptive debiasing operator $\tilde{\mathbf{x}} = \text{normalize}(\mathbf{x} - \mathbf{U}(\boldsymbol{\lambda} \odot \mathbf{U}^\top \mathbf{x}))$.
- Compare five adaptive mechanisms: fixed weighted attenuation, data-free spectral Wiener attenuation, calibrated MMSE gains, image-dependent scaling, and token-dependent spatial template subtraction.
- Benchmark all strategies against the hard-projection baseline using the same datasets and protocols for both segmentation (mIoU) and semantic correspondence (SPair-71k PCK).

---

## 2. Multi-Prototype Reference Matching

### Motivation
Standard INSID3 summarizes the annotated reference mask into a single global mean prototype vector. A single vector can under-represent objects composed of multiple distinct visual parts, textures, or sub-regions.

### Objectives
- Cluster DINOv3 patch features inside the annotated reference mask into multiple complementary prototypes (via spherical $k$-means and hierarchical agglomerative clustering).
- Extend region matching so target candidate regions are scored against the full prototype set.
- Evaluate multiple aggregation and many-to-many matching rules: maximum similarity, area-weighted averaging, log-sum-exp (LSE), symmetric Chamfer similarity, and unbalanced entropic Optimal Transport (Sinkhorn).
- Quantify the effect of prototype count $M$ and downstream threshold interactions (`merge_threshold` $\alpha$) using identical mIoU evaluation protocols.

---

## Reference
[1] Claudia Cuttano, Gabriele Trivigno, Christoph Reich, Daniel Cremers, Carlo Masone, and Stefan Roth. INSID3: Training-free in-context segmentation with DINOv3. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2026.
