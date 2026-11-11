# 4. How the experiments were done

## 4.1 Rules we followed

1. **Never tune on test episodes.** Every choice (mode, strength, number of prototypes, …) is made on a
   separate *tuning split* of COCO train2014 images. The official test episodes are only used to check the
   chosen settings.
2. **Fix the selection rule before looking.** The setting with the highest tuning mIoU wins (ties: the
   earlier name). `scripts/select_best.py` does this automatically.
3. **Compare on the same episodes.** All methods see the same episodes (same seed, same order; checked with
   a fingerprint of every episode). Differences are judged with a paired bootstrap (4.4).
4. **Same machine.** Different GPUs change individual episodes (4.5), so a method is only compared with a
   baseline computed on the same machine.
5. **Report everything,** including negative results.

## 4.2 Data

| | Images | Episodes | Used for |
|---|---|---|---|
| COCO tuning split | COCO train2014 (`splits/trn`), all 80 classes | 1,000 (round 1), 4,000 (round 2); 1,000 for 5-shot | choosing settings |
| COCO-20ⁱ test | val2014, 4 folds of 20 classes | 1,000 per fold | final check |
| PASCAL-Part test | 4 folds (animals, indoor, person, vehicles) | up to 2,500 per fold | final check (parts) |
| SPair-71k test | 18 categories | 12,234 image pairs | correspondence |
| calibration | 500 random COCO train2014 images (no labels) | — | measuring the positional components |

The tuning split shares no image with the test folds; the run log prints the count (0).

Why 4,000 tuning episodes in round 2: with 1,000 episodes a COCO fold varies by about ±1.3 mIoU between
random episode sets (PFENet, Tab. 10), more than the differences we were looking for. Four times the
episodes halves the noise.

## 4.3 What was run

**Round 1** (`STAGE=r1_tune, r1_combined, r1_test, r1_corr` in `run_experiments.sh`):

* Task 1, 29 settings: hard with s = 100/300/700/900, none, weighted (4 α), spectral (3 knees × 2 sharpness),
  calibrated (4 β), image (2 sources × 3 β), token (2 gains × 2 β).
* Task 2, 42 settings: max / weighted / lse with M = 2…8, lse temperature, Chamfer, optimal transport
  (3 ε × 4 ρ), agglomerative grouping, extra global prototype, grouping in debiased space, higher α.
* 5 combined settings. Then test of the selected Task 1, Task 2 and combined settings (1-shot; combined also
  5-shot) and 7 SPair runs (README setting ViT-B/768, s = 20).

**Round 2** (`STAGE=bias, t1, t2, corr, attribution, consistency, report`):

| stage | question |
|---|---|
| `bias` | how much same-position bias does every variant leave, and how much of the image's own structure does it keep? |
| `t1` | where does the benefit of the hard projection come from (controls), and does the round-1 winner survive 4,000 episodes? |
| `t2` | does a lower α, a score rescaling or an area-adaptive M help; per-shot vs pooled prototypes with 5 shots |
| `corr` | Task-1 variants on SPair with the segmentation backbone (ViT-L/1024, s = 500) |
| `attribution` | which task caused the 5-shot drop of round 1? |
| `consistency` | can runs from two machines be compared episode by episode? |

## 4.4 How differences are judged (paired bootstrap, `scripts/compare_runs.py`)

For two methods A and B on the same episodes:

1. For every fold, draw that fold's episodes *with replacement* (the same draw for A and B, which is what
   "paired" means).
2. Compute the class-wise mIoU of A and of B on this resample (the official formula), and average over the
   folds.
3. Store B − A. Repeat 5,000 times.
4. The 95 % confidence interval is the 2.5 % and 97.5 % points of the stored differences. If it does not
   contain 0, the difference is larger than the episode sampling noise.

For SPair the same is done with the per-pair PCK.

Two cautions:
* When many settings are compared and the best is picked, its lead is partly luck. So a tuning "win" must be
  confirmed on new episodes.
* A confidence interval only covers the randomness of the episodes, not other choices (benchmark, seed of
  the basis, thresholds).

## 4.5 Same configuration on two machines (`scripts/check_consistency.py`)

The baseline on COCO fold 0 was run on both machines. The rule, fixed in advance: interchangeable if the
mIoU differs by ≤ 0.05 *and* ≥ 98 % of the episodes are identical.

Result: −0.03 mIoU, but only 34.6 % identical episodes, so they are not interchangeable. Tiny
floating-point differences (GPU type, bf16/TF32 kernels) flip some clustering and threshold decisions.
Hence every round-2 stage computes its own baseline.

## 4.6 What is saved for every run

```
<output_root>/<experiment>/<dataset>[-tune]/fold<f>_shot<k>_seed<s>/
    config.yaml       the full configuration
    env.json          versions, GPU, host, git commit
    log.txt           the command and progress
    episodes.jsonl    one line per episode: fingerprint, class, intersection, union, IoU, time, diagnostics
    metrics.json      mIoU, FB-IoU, number of episodes, time, memory
```

`episodes.jsonl` makes every number in `docs/5_results.md` checkable, and lets an interrupted run resume.
