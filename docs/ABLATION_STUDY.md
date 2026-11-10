# Ablation study design

All ablations run on COCO tuning episodes (EXPERIMENTAL_PROTOCOL.md §3) with bilinear upsampling;
only the selected configurations go to the test benchmarks. Observed tables: `results/`.
Each row names the question it answers, so a result can be interpreted without re-reading the code.

## Task 1

| Axis | Values | Question | Controls |
|---|---|---|---|
| hard rank s | 0 (none), 100, 300, 500, 700, 900 | Does our data reproduce the rise-then-fall of paper Fig. 7? (if not, later comparisons are suspect) | everything else official |
| weighted α | 0.25, 0.5, 0.75, 0.9, 1 | Is partial uniform removal better than full removal? | rank = 500 |
| spectral knee × γ | {300, 500, 700} × {1, 4} | Does a soft rank cut help without any data? γ = 4 approaches the hard cut | full rank |
| calibrated β | 0.5, 1, 1.5, 2 | Is the MMSE point (β = 1) best, or is more/less suppression better downstream? | full rank, same calibration file |
| image source × β | {calibration, noise} × {0.5, 1, 2} | Does per-image adaptation help; is the noise image a usable positional-energy estimate? | |
| token gain × β | {least-squares, fixed} × {0.5, 1} | Does a position-specific template beat component gains? Does the token amplitude matter? | γ_max = 2 |
| correspondence | 6 modes on SPair-71k | Which mode gives the most reliable cross-image matches? (isolates matching from segmentation) | ViT-B 768 s = 20 as README |

Interpretation guide: if `calibrated β=1` ≈ `hard` but `weighted α<1` < `hard`, the hard cut is near-optimal
for this pipeline; if spectral/calibrated beat hard on SPair but not on mIoU, the bottleneck is downstream
(clustering/threshold), not matching.

## Task 2

| Axis | Values | Question |
|---|---|---|
| M | 1 (check), 2, 3, 4, 6, 8 | How many prototypes help; where does fragmentation start? |
| aggregation | max, weighted, lse(T) | Non-linear vs linear pooling (MATH §3.3) |
| lse T | 0.02, 0.05, 0.1 | Position between weighted and max |
| region matching | prototype, Chamfer, OT(ε, ρ) | Does many-to-many region scoring help seed selection / aggregation? |
| OT ρ | 0.1, 0.5, 2, 100 (≈ balanced) | Partial vs full coverage of the reference |
| clustering | k-means vs agglomerative (ref_tau 0.5/0.6/0.7) | Fixed vs adaptive prototype count |
| global prototype | with / without | Does keeping the mean protect against bad clusters? |
| grouping space | original vs debiased | INSID3 principle (group in original space) applied to the reference |
| α | 0.2, 0.25, 0.3, 0.35 | Threshold–score-scale confound (MATH §3.5) |

## Combined

| Run | Question |
|---|---|
| `c_selected` | Are Task 1 and Task 2 additive? (compare with sel. Task 1 / Task 2 tune runs) |
| `c_hard_debias` | Task 2 alone in the combined config (sanity vs `t2_*`) |
| `c_group_debiased` | Phase-5 question: should reference grouping use debiased features once debiasing is adaptive? |
| `c_alpha{0.15,0.25}` | Threshold robustness of the combination |

## Reporting template

For each axis: plot tuning mIoU vs the value (with the baseline as a horizontal line) and give the paired
bootstrap CI of the best value vs baseline. Use `results/summary_segmentation_runs.csv` as the data source.
