# Failure analysis

This document catalogues candidate failure modes and summarizes the episode-level failure diagnostics generated on COCO `train2014` tuning runs (`results/fail_*.csv`).

## Tools

* `scripts/analyze_failures.py --a <baseline run> --b <method run>`: per-episode ΔIoU, wins/losses,
  ΔIoU by GT-area tercile, by class, and whether the method predicts larger masks.
* Per-episode diagnostics in `episodes.jsonl`: `fg_iou`, `pred_area`, `gt_area`, `num_prototypes`,
  `candidate_tokens`, `num_clusters`, `seed_cluster`, `seed_score`, `debias_stats` (mean λ or γ),
  `time_ms`, `peak_mem_mb`.
* `run.save_predictions=N` stores masks and overlay figures for the first N episodes.
* `scripts/visualize.py similarity|prototypes` for single episodes.

## Anticipated failure modes

| # | Symptom | Likely cause | Check |
|---|---|---|---|
| F1 | Task-1 mode loses mIoU, PCK unchanged | too little positional removal → spurious positional matches in background | `debias_stats.mean_lambda`; similarity maps show coordinate-aligned blobs; try β > 1 |
| F2 | Task-1 mode raises PCK but not mIoU | matching improved, bottleneck is clustering / threshold | compare `candidate_tokens`, `seed_score` distributions |
| F3 | `image` mode with `noise` variance ≈ hard | constant-image energy over-estimates positional energy → λ clamps at 1 | mean λ close to 1 |
| F4 | `token` mode unstable | templates not stationary across images, LS gain saturates at γ_max | `debias_stats.mean_gamma` near γ_max |
| F5 | Task 2 over-segments (pred_area ≫ gt_area) | max/lse scores larger → more clusters above α | `analyze_failures` "larger mask" line; α sweep (E2.7) |
| F6 | Task 2 picks a wrong seed on distractors | a minority prototype (background leaked into the mask, boundary tokens) matches other objects | `seed_score`; `include_global=true`; `min_cluster_fraction` |
| F7 | No effect on part benchmarks | part masks are small → `num_prototypes == 1` (M_eff collapse) | distribution of `num_prototypes` |
| F8 | OT worse than prototype matching | balanced coverage (large ρ) penalises part-level clusters | ρ sweep; expect small ρ to win |
| F9 | Reference grouped by position | grouping features still positional → prototypes = spatial halves | prototype visualisation; `c_group_debiased` |
| F10 | Baseline far from published numbers | wrong weights / resolution / CRF off / data layout | EXPERIMENTAL_PROTOCOL.md §6, TROUBLESHOOTING.md |

## Observed Findings (COCO Tuning Split, 1,000 Paired Episodes)

* **`fail_none_vs_baseline.csv` (`t1_none` vs `baseline`):** Removing debiasing degrades mIoU by $-4.34\%$ ($p=0.000$); the mean episode IoU delta from `baseline` over `t1_none` is $+3.08\%$ overall, driven by all size terciles ($+2.41\%$ small, $+2.61\%$ medium, $+4.22\%$ large).
* **`fail_spectral_k500_g4.csv` (`t1_spectral_k500_g4` vs `baseline`):** Spectral Wiener attenuation yields a mean foreground IoU delta of $+0.48\%$ across episodes, concentrated in the **small GT-area tercile** (`gt_area <= 26,530` px at $1024^2$, $n=334$: **$+0.98\%$**, $48.09\%$ vs $47.11\%$), vs $+0.36\%$ medium and $+0.11\%$ large.
* **`fail_t2_lse_m2.csv` (`t2_lse_m2_t0.05` vs `baseline`):** Smooth LSE pooling improves the **medium GT-area tercile** ($n=333$) by **$+0.60\%$** ($60.57\%$ vs $59.97\%$), but drops $-0.68\%$ on small objects and $-0.44\%$ on large objects.
* **F5 Confirmed (Threshold Confound):** Increasing $M$ to $4$ without recalibrating $\alpha=0.20$ lowers mIoU to $55.94\%$ (`max`) / $56.69\%$ (`lse`); naively raising $\alpha \in \{0.25, 0.30, 0.35\}$ drops recall steeply (`54.02%` $\rightarrow$ `52.17%` $\rightarrow$ `49.20%` for `max`).
