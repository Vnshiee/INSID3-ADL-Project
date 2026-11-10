# Experiments

Status: **EXECUTED** for calibration, 77 COCO `train2014` tuning runs, 7 SPair-71k correspondence evaluations, and all 4-fold 1-shot/5-shot test benchmarks on COCO-20ⁱ and PASCAL-Part (`DATASETS="coco pascal_part"` on RTX 3070 8 GB; see [RESULTS.md](RESULTS.md)).
Commands assume `env.sh` is configured (SERVER_SETUP.md). All runs are resumable.

## E0 — Environment and baseline reproduction

| ID | Command | Purpose |
|---|---|---|
| E0.1 | `python scripts/check_env.py --datasets coco spair` | pre-flight |
| E0.2 | `STAGE=test ./run_baseline.sh` | official baseline on `$DATASETS` + no-debias + SPair (original / debiased) |
| E0.3 | `CRF=1 STAGE=test ./run_baseline.sh` | Tab. 1 setting |
| E0.4 | `STAGE=tune ./run_baseline.sh` | baseline on tuning episodes (reference for selection) |

## E1 — Task 1 (adaptive positional debiasing)

| ID | Stage | Runs | Variable(s) |
|---|---|---|---|
| E1.1 | `STAGE=calibrate ./run_task1.sh` | 2 calibrations | ViT-L@1024 (s=500), ViT-B@768 (s=20) |
| E1.2 | tune | `t1_none`, `t1_hard_s{100,300,700,900}` (+ baseline s=500) | rank of the hard projection (reproduces Fig. 7 trend on tuning data) |
| E1.3 | tune | `t1_weighted_a{0.25,0.5,0.75,0.9}` | fixed weighted projection |
| E1.4 | tune | `t1_spectral_k{300,500,700}_g{1,4}` | data-free component-wise attenuation |
| E1.5 | tune | `t1_calibrated_b{0.5,1.0,1.5,2.0}` | MMSE component-wise attenuation |
| E1.6 | tune | `t1_image_{cal,noise}_b{0.5,1.0,2.0}` | image-dependent correction |
| E1.7 | tune | `t1_token_{ls,fixed}_b{0.5,1.0}` | token-dependent correction |
| E1.8 | `STAGE=corr ./run_task1.sh` | `corr_base768_{hard,weighted_a0.5,spectral_k20_g1,calibrated_b1,image_b1,token_ls}` | correspondence quality per mode |
| E1.9 | `STAGE=test ./run_task1.sh` | `sel_task1` on all benchmarks | confirmatory |

## E2 — Task 2 (multi-prototype matching)

| ID | Runs | Variable(s) |
|---|---|---|
| E2.0 | `t2_check_m1` | sanity: must equal baseline |
| E2.1 | `t2_{max,weighted}_m{2,3,4,6,8}`, `t2_lse_m{2,3,4,6,8}_t0.05` | prototype count × aggregation |
| E2.2 | `t2_lse_m4_t{0.02,0.1}` | lse temperature |
| E2.3 | `t2_chamfer_m4` | many-to-many (Chamfer) |
| E2.4 | `t2_ot_m4_e{0.02,0.05,0.1}_r{0.1,0.5,2.0,100.0}` | many-to-many (unbalanced OT): ε × ρ |
| E2.5 | `t2_agglo_tau{0.5,0.6,0.7}_max` | adaptive prototype count |
| E2.6 | `t2_max_m4_global`, `t2_max_m4_groupdeb` | keep global prototype; grouping space |
| E2.7 | `t2_{max,lse}_m4_alpha{0.25,0.3,0.35}` | merge-threshold confound |
| E2.8 | `sel_task2` on all benchmarks (`STAGE=test ./run_task2.sh`) | confirmatory |

## E3 — Combined

| ID | Runs | Purpose |
|---|---|---|
| E3.1 | `c_selected`, `c_group_debiased`, `c_hard_debias`, `c_alpha{0.15,0.25}` (tune) | integration choices |
| E3.2 | `sel_combined` 1-shot on all benchmarks, 5-shot on `$DATASETS_5SHOT` (+ 5-shot baseline) | confirmatory |

## E4 — Analysis (CPU, after E1–E3)

| ID | Command |
|---|---|
| E4.1 | `./evaluate.sh` → `results/summary.md`, `results/comparisons.md` |
| E4.2 | `python scripts/analyze_failures.py --a <baseline run> --b <method run> --out results/fail_<name>` |
| E4.3 | `python scripts/visualize.py similarity …`, `… prototypes …` on chosen episodes |
| E4.4 | runtime/memory: columns `ms/episode`, `peak MB` of `results/summary.md` (run `STAGE=timing` alone on the GPU) |

## Measured Runtime on NVIDIA GeForce RTX 3070 (8 GB VRAM)

| Block | Volume | Measured Runtime |
|---|---|---|
| Calibration (ViT-L@1024 & ViT-B@768) | 500 images × 2 | ~2 min total (~97s ViT-L, ~26s ViT-B) |
| Tuning sweeps (77 runs × 1,000 eps, 1-shot) | 77,000 eps | ~545–555 ms/episode (~9.7 min/run; ~12.5 h total) |
| Test benchmarks (COCO 4,000 eps/run, PASCAL-Part 8,459 eps/run) | 87,213 eps | ~541–551 ms/ep (1-shot); ~1,190–1,205 ms/ep (5-shot) |
| SPair-71k (12,234 test pairs, ViT-B 768) | 7 runs (85,638 pairs) | ~16 min per run (~1.9 h total) |

Clustering runs on the CPU (scikit-learn); a slow CPU can dominate. Several runs can share one GPU
(memory permitting) by launching them in parallel with different `run.exp_name`.
