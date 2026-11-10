# Limitations

## Execution Scope

1. **Completed GPU evaluation scope.** All calibration, 77 COCO `train2014` tuning sweeps (77,000 episodes), 7 SPair-71k semantic correspondence runs (12,234 test pairs each), and 14 full 4-fold test evaluations across COCO-20ⁱ and PASCAL-Part (1-shot and 5-shot; 87,213 episodes) were executed on an NVIDIA GeForce RTX 3070 GPU (8 GB VRAM, CUDA 12.6, PyTorch 2.7.1) using bilinear upsampling.
2. **Untested domains & CRF.** DenseCRF post-processing (`model.mask_refiner=crf`) and the remaining 7 auxiliary domain benchmarks listed in `EXPERIMENTAL_PROTOCOL.md` (LVIS-92ⁱ, PACO-Part, ISIC 2018, Chest X-ray, SUIM, iSAID-5ⁱ, PerMIS) were not executed due to single-GPU compute constraints; baseline Tab. 1 paper comparisons therefore reflect the bilinear pipeline (`56.13%` COCO-20ⁱ test, `49.13%` PASCAL-Part test).
3. **Unit test suite.** The 126 automated unit tests in `tests/` use a deterministic synthetic encoder (`tests/conftest.py`) to verify tensor shapes, algebraic identities, invariants, CLI plumbing, and RNG isolation independently of weights/data.
4. **SPair-71k statistical CIs.** `scripts/compare_all.py` executes fold-stratified paired bootstrap intervals for segmentation runs only; round-1 SPair-71k results report point PCK over all 12,234 pairs.

## Of the methods

5. **Linear, per-component corrections.** Task-1 modes re-weight coefficients in INSID3's basis. Positional
   information that is not linearly separable in that basis is not addressed.
6. **Additive-model assumption.** ρ̂ attributes any position–content correlation in natural images (sky at
   the top, ground at the bottom) to "position". Such correlations are dataset priors, partly semantic.
7. **Calibration dependence.** ρ̂ and templates are estimated on COCO train2014 and are specific to one
   (backbone, resolution). They may not transfer to medical / aerial / underwater domains. COCO-2017-based
   benchmarks (LVIS, PACO) share images with COCO-2014 train (labels are never used).
8. **Resolution coupling.** Templates exist only for the calibration grid; `token` mode cannot run at
   another resolution without recalibration (an error is raised).
9. **Fixed thresholds.** α = 0.2 and the forward-mask threshold 0 were tuned for the single-prototype score;
   multi-prototype scores have a different scale. We tune α on tuning episodes, which adds a degree of freedom
   compared to the original method.
10. **Prototype count.** k-means needs M; small masks collapse to one prototype by design, so Task 2 cannot
    help very small references. Agglomerative reference clustering has its own threshold.
11. **Hyper-parameter budget.** Our sweeps evaluate many more configurations than the original method had;
    selection on tuning episodes limits but does not remove optimistic bias. Test results of the selected
    configurations are the only confirmatory evidence.
12. **Many-to-many OT** adds ε, ρ and iterations; balanced OT conflicts with INSID3's part-level clusters.
13. **Single seed.** Episodes are drawn with seed 0 (official). The method is deterministic given the
    episode, so seed variation only changes the episode sample; additional seeds (`run.seed`) quantify this.

## Of the protocol

14. Paper Tab. 1 uses CRF; our ablations use bilinear upsampling for cost reasons. Final comparisons should
    be repeated with CRF if it builds.
15. The tuning split exists only for COCO; other domains are evaluated zero-shot with COCO-selected settings
    (as in the paper).
