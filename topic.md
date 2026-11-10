# 5 INSID3: Training-Free In-Context Segmentation with DINOv3

**Code:** https://github.com/visinf/INSID3

### Summary
INSID3 [5] is a training-free in-context segmentation method based on a frozen DINOv3 backbone. The method first removes positional bias from the extracted features, decomposes the target image into coherent candidate regions, and matches these regions with an annotated reference example using semantic correspondence and reference prototypes. This enables segmentation without fine-tuning, an additional decoder, or an auxiliary segmentation model.

---

### 5.1 Task 1: Adaptive Positional Debiasing

#### 5.1.1 Description
INSID3 identifies a low-dimensional positional-bias subspace in DINOv3 features and removes this information by projecting the features onto its orthogonal complement. However, applying the same hard projection to all features may also suppress semantic information that overlaps with the estimated positional directions. You are required to introduce **Adaptive Positional Debiasing**, where positional components are suppressed according to their estimated contribution rather than being removed uniformly.

#### 5.1.2 Instructions
- Analyze the positional components identified by INSID3 and measure their contribution across different images and feature tokens.
- Replace the hard orthogonal-complement projection with a soft debiasing mechanism that adaptively controls the amount of positional information removed.
- Investigate different strategies, such as weighted projection, component-wise attenuation, or image-dependent positional correction.
- Evaluate the modified method using the same datasets and experimental protocol as the original INSID3 implementation.
- Compare the proposed method with the original positional-debiasing approach in terms of segmentation mIoU and semantic correspondence quality.

---

### 5.2 Task 2: Multi-Prototype Reference Matching

#### 5.2.1 Description
INSID3 represents the annotated reference region using a compact semantic representation for matching candidate regions in the target image. A single reference representation may be insufficient when the annotated region contains multiple visual patterns, textures, or semantic parts. You are required to introduce **Multi-Prototype Reference Matching**, where the reference region is represented using multiple complementary prototypes instead of a single representation.

#### 5.2.2 Instructions
- Cluster the DINOv3 patch features within the annotated reference region to obtain multiple reference prototypes.
- Modify the region-matching stage so that candidate target regions are compared against the complete set of reference prototypes.
- Investigate different prototype aggregation strategies, such as maximum similarity, weighted similarity, or many-to-many matching.
- Evaluate different numbers of reference prototypes and analyze their effect on segmentation performance.
- Compare the proposed matching strategy with the original INSID3 matching mechanism using the same segmentation benchmarks and mIoU evaluation protocol.

---

### Reference
[5] Claudia Cuttano, Gabriele Trivigno, Christoph Reich, Daniel Cremers, Carlo Masone, and Stefan Roth. INSID3: Training-free in-context segmentation with DINOv3. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2026.

