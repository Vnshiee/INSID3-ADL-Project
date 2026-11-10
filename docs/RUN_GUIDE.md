# Running the revision experiments — step by step

All commands run in the repository root. Every stage is resumable: if a run is interrupted, start the same
command again; finished episodes are skipped.

**Phase 1** covers the core benchmarks (COCO-20ⁱ, PASCAL-Part, SPair-71k); conclusions are written from it.
**Phase 2** (CRF, further benchmarks, extra seeds, runtime) is added later; its results are appended to the
same tables. Round-1 results stay as they are. Every stage computes its own baseline, so a method is always
compared with a baseline from the same machine.

## Step 1 — Environment

```bash
cp env.example.sh env.sh
```

Edit `env.sh`: data, output and weight folders, the Python interpreter, and for the revision

```bash
export INSID3_CALIBRATION_PATH=$INSID3_OUTPUT_ROOT/calibration/calib_large_1024_v2.pt
export INSID3_CALIBRATION_PATH_BASE768=$INSID3_OUTPUT_ROOT/calibration/calib_base_768_v2.pt
export DATASETS="coco pascal_part"
```

Python environment and checks (`--with-crf` only for the phase-2 CRF stage; it needs `nvcc`):

```bash
./setup_server.sh
```

Datasets: `./prepare_datasets.sh --yes coco pascal_part spair`, then

```bash
python scripts/check_env.py --datasets coco pascal_part spair
```

## Step 2 — Sanity (≈ 15 min)

```bash
STAGE=sanity ./run_revision.sh
```

Runs the tests, checks that the official files are unchanged, checks the data and makes a 20-episode smoke
run (`outputs/smoke/`). Its mean `time_ms` in `metrics.json` estimates the duration of the other stages.

## Step 3 — Calibration and direct bias analysis (≈ 20 min)

```bash
STAGE=calibrate ./run_revision.sh && STAGE=bias ./run_revision.sh
```

Read `results/positional_bias.md` and the figures in `outputs/calibration/figs_*`.

## Step 4 — Phase-1 experiments

```bash
for s in t1 t2 corr attribution; do STAGE=$s ./run_revision.sh; done
```

| Stage | Content |
|---|---|
| `t1` | Task-1 controls (small s, ρ-threshold, random, PCA, centring, eval-mode basis) and re-checks, 4,000 COCO tuning episodes |
| `t2` | Task-2: lower α, score calibration, adaptive M, 5-shot per-shot vs pooled |
| `corr` | SPair-71k with the segmentation backbone (ViT-L, 1024 px) for the key Task-1 variants |
| `attribution` | 5-shot test of the round-1 Task-1 and Task-2 selections |

## Step 5 — Selection and report (CPU only)

```bash
STAGE=select ./run_revision.sh && STAGE=report ./run_revision.sh
```

Files in `results/`:

| File | Answers |
|---|---|
| `positional_bias.md` | how much same-position bias each Task-1 variant leaves, how much structure it keeps |
| `comparisons.md` (rows `t1_*`, coco-tune) | is the projection's effect in the first few components? random/PCA/centring controls; does the round-1 winner survive 4× more episodes? |
| `comparisons.md` (SPair section, `corr_large1024_*`) | Task-1 variants on correspondence at the segmentation backbone, with CIs |
| `comparisons.md` (rows `t2_*`) | lower α, score calibration, adaptive M; 5-shot pooled vs per-shot |
| `comparisons.md` (rows `sel_task1/2`, shots 5) | which task causes the 5-shot drop of `sel_combined` |
| `fail_*_5shot_pascal_f*.csv` | where the 5-shot PASCAL-Part losses come from |
| `revision_selection_report.md` | what round 2 would select (exploratory) |
| `consistency.json` | the same baseline on two different GPUs: same mean, but many individual episodes differ |

Write the conclusions from these files.

---

## Phase 2 — later, one block at a time

Each block appends rows to the same tables; re-run `STAGE=report` afterwards.

| Block | Command | Notes |
|---|---|---|
| CRF (paper Tab. 1 setting) | `./setup_server.sh --with-crf` then `STAGE=crf ./run_revision.sh` | needs `nvcc` |
| Smaller benchmarks | `./prepare_datasets.sh --yes isic lung suim isaid` then `STAGE=bench ./run_revision.sh` | baseline + round-1 selections |
| Large benchmarks | `./prepare_datasets.sh --yes lvis paco_part permis` then `BENCH_DATASETS="lvis paco_part permis" STAGE=bench ./run_revision.sh` | ≈ 40 GB data, LVIS has 10 folds |
| Extra episode seeds | `STAGE=seeds ./run_revision.sh` | COCO seeds 1–2; PFENet shows 1,000-episode folds vary by ±1.3 |
| Round-2 selections on test | `STAGE=test_rev ./run_revision.sh` | only if `revision_selection_report.md` picks something new; report as exploratory |
| Runtime table | `STAGE=timing ./run_revision.sh` | run with nothing else on the GPU |
