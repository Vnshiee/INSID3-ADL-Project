# Research plan

## Goal

Two training-free extensions of INSID3 (topic.md), evaluated with the original protocols:

* **Task 1 — Adaptive positional debiasing:** replace the hard orthogonal-complement projection by a
  suppression that removes positional information according to its measured contribution.
* **Task 2 — Multi-prototype reference matching:** replace the single reference prototype by several
  prototypes obtained by clustering DINOv3 patch features inside the reference mask.

## Research questions

* RQ1 How positional are the 500 components INSID3 removes, across images and tokens? (Module A)
* RQ2 Does partial / adaptive removal improve cross-image matching (PCK) and segmentation (mIoU)? (H1)
* RQ3 Does a multi-prototype reference improve matching of heterogeneous regions, and which aggregation
  works best? (H2, H3)
* RQ4 Are both changes complementary? (H4)

## Plan and status

| Phase | Content | Status |
|---|---|---|
| 0 | Repository/paper audit, resource inventory | **done** (REPOSITORY_AUDIT.md, RESOURCE_REQUIREMENTS.md) |
| 1 | Pipeline trace with shapes; locate modification points | **done** (REPOSITORY_AUDIT.md §2–4) |
| 2 | Baseline preserved byte-identically; resumable runner; configs | **done**, equivalence tested on CPU |
| 3 | Task 1: analysis + 5 adaptive modes | **done** (implemented, tested, evaluated on COCO + SPair) |
| 4 | Task 2: prototypes + 4 aggregation / 3 region rules | **done** (implemented, tested, evaluated on COCO + PASCAL-Part) |
| 5 | Integration, feature-space choices | **done** |
| 6 | Experiment framework (tuning split, selection, bootstrap, failure analysis) | **done** |
| 7 | Workflow scripts (setup, weights, data, run scripts) | **done & executed** |
| 8 | Tests | 126 tests **executed and passing** |
| 9 | Documentation | **done** |
| 10 | Static & results audit | **done** (PROJECT_STATE.md) |
| S1–S5 | GPU execution: calibration → 77 tune sweeps → SPair → test (COCO, PASCAL-Part) → analysis | **done** ([RESULTS.md](RESULTS.md)) |

## Decision points (pre-registered)

1. If the baseline reproduction check fails (EXPERIMENTAL_PROTOCOL §6), stop and fix the environment.
2. If `t2_check_m1` ≠ baseline, stop (implementation error).
3. Mode / hyper-parameter selection only by `scripts/select_best.py` on tuning episodes.
4. Report all test benchmarks, including losses; claim an improvement only with a paired CI excluding 0.
5. If no Task-1 mode beats hard projection on tuning data, report this as a negative result: the
   evidence then supports the original design choice (still a valid outcome of the investigation).

## Design Defaults

* Tuning on COCO train2014 episodes (mirrors the paper's own selection protocol).
* Calibration: 500 random COCO train2014 images, seed 0.
* Ablations with bilinear upsampling; CRF only for the final comparison.
* Prototypes are grouped in the original feature space and represented in the debiased space.
* Many-to-many matching via Chamfer and unbalanced entropic OT.
