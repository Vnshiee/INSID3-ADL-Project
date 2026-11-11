#!/usr/bin/env bash
# All experiments of this project, as stages.   Usage:   STAGE=<name> ./run_experiments.sh
#
# Every stage is resumable: if it is interrupted, start the same command again (finished episodes are
# skipped). Each stage computes its own baseline, so a method is always compared with a baseline from the
# same machine (different GPUs change individual episodes slightly).
#
# SETUP (once)
#   sanity        tests, official files unchanged, data present, 20-episode smoke run
#   calibrate     measure the positional components on 500 COCO train2014 images (ViT-L/1024 and ViT-B/768)
#
# ROUND 1 — first sweeps (1,000 COCO tuning episodes), selection, test benchmarks  -> outputs/, results/round1/
#   r1_tune       Task-1 and Task-2 sweeps on the tuning split, then selection (configs/selected/)
#   r1_combined   Task 1 + Task 2 together on the tuning split, then selection
#   r1_test       baseline, no debiasing and the three selections on COCO-20i and PASCAL-Part (1-shot; 5-shot)
#   r1_corr       SPair-71k, README setting (ViT-B, 768 px, s = 20)
#   r1_report     tables + paired bootstrap -> results/round1/
#
# ROUND 2 — checks of round 1 with controls and 4,000 tuning episodes   -> outputs/round2/, results/round2/
#   bias          residual positional bias of every debiasing variant (no segmentation, minutes)
#   t1            Task-1 controls: small s, rho threshold, random, PCA, centring, eval-mode basis; re-checks
#   t2            Task-2 checks: lower alpha, score calibration, adaptive M, 5-shot per-shot vs pooled
#   corr          SPair-71k with the segmentation backbone (ViT-L, 1024 px, s = 500)
#   attribution   5-shot test of the round-1 Task-1 and Task-2 selections (which task causes the 5-shot drop?)
#   consistency   the same baseline run on two machines, compared episode by episode (set INSID3_R1_ROOT)
#   report        selection + tables + paired bootstrap -> configs/selected_rev/, results/round2/
#
# LATER (round 2, phase 2)
#   seeds         COCO episode seeds 1 and 2 for the baseline and the round-1 selections
#   crf           CRF mask refinement (paper Tab. 1 setting; needs ./setup_server.sh --with-crf)
#   bench         further benchmarks:  BENCH_DATASETS="isic lung suim isaid" STAGE=bench ./run_experiments.sh
#   timing        runtime and memory, 200 episodes per method, with nothing else running on the GPU

set -euo pipefail                      # stop at the first error or unset variable
cd "$(dirname "$0")"
if [[ -f env.sh ]]; then source env.sh; else source env.example.sh; fi
PY="${INSID3_PYTHON:-python}"
STAGE="${STAGE:-}"
OUT="$INSID3_OUTPUT_ROOT"              # round 1 results live directly here
OUT2="$INSID3_OUTPUT_ROOT/round2"      # round 2 results live in their own folder
CAL="debias.calibration_path=$INSID3_CALIBRATION_PATH"
export DATASETS="${DATASETS:-coco pascal_part}"
mkdir -p "$OUT/logs"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/logs/run_experiments.log"; }

require_cuda() {
  "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "ERROR: CUDA is not available to $PY (run scripts/check_env.py)."; exit 1; }
}

# tune <config> <exp_name> <episodes> <output root> [more --set overrides]
#   one run on the COCO tuning split (train2014 images, never used by the test benchmarks)
tune() {
  local cfg="$1" name="$2" episodes="$3" root="$4"; shift 4
  log "tune $name"
  "$PY" scripts/run_segmentation.py --config "$cfg" --set data.split=tune data.dataset=coco \
      data.tune_episodes="$episodes" run.output_root="$root" run.exp_name="$name" run.full_diagnostics=true "$@"
}

# test <config> <exp_name> <output root> [more --set overrides]
#   all folds of every benchmark in $DATASETS (official episodes)
test_all() {
  local cfg="$1" name="$2" root="$3"; shift 3
  for ds in $DATASETS; do
    log "test $name on $ds"
    "$PY" scripts/run_segmentation.py --config "$cfg" --folds all \
        --set data.dataset="$ds" run.output_root="$root" run.exp_name="$name" run.full_diagnostics=true "$@"
  done
}

# corr <config> <exp_name> <output root> [more --set overrides]   (SPair-71k keypoint matching)
corr() {
  local cfg="$1" name="$2" root="$3"; shift 3
  log "corr $name"
  "$PY" scripts/run_correspondence.py --config "$cfg" --set run.output_root="$root" run.exp_name="$name" "$@"
}

# report <output root> <results folder>   (CPU only)
report() {
  mkdir -p "$2"
  "$PY" scripts/aggregate_results.py --root "$1" --out "$2/summary"
  "$PY" scripts/compare_all.py --root "$1" --reference baseline --out "$2/comparisons"
}

case "$STAGE" in
# ───────────────────────────── setup ─────────────────────────────
sanity)
  "$PY" -m pytest tests -q
  "$PY" scripts/verify_upstream.py
  "$PY" scripts/check_env.py --datasets coco pascal_part spair
  require_cuda
  "$PY" scripts/run_segmentation.py --config configs/baseline.yaml \
      --set run.max_episodes=20 run.exp_name=smoke run.save_predictions=5
  ;;
calibrate)
  require_cuda
  "$PY" scripts/calibrate_positional.py --config configs/base.yaml --image-dir "$INSID3_CALIB_IMAGES" \
      --num-images "$INSID3_CALIB_NUM" --out "$INSID3_CALIBRATION_PATH"
  "$PY" scripts/calibrate_positional.py --config configs/base.yaml --image-dir "$INSID3_CALIB_IMAGES" \
      --num-images "$INSID3_CALIB_NUM" --out "$INSID3_CALIBRATION_PATH_BASE768" \
      --set model.model_size=base model.image_size=768 model.svd_components=20
  for f in "$INSID3_CALIBRATION_PATH" "$INSID3_CALIBRATION_PATH_BASE768"; do
    "$PY" scripts/visualize.py calibration --calibration "$f" --out "$(dirname "$f")/figs_$(basename "$f" .pt)"
  done
  ;;

# ───────────────────────────── round 1 ─────────────────────────────
r1_tune)
  require_cuda
  E=1000
  tune configs/baseline.yaml baseline $E "$OUT"
  # Task 1: how much of each positional direction to remove
  tune configs/baseline_no_debias.yaml t1_none $E "$OUT"
  for s in 100 300 700 900; do tune configs/baseline.yaml t1_hard_s$s $E "$OUT" model.svd_components=$s; done
  for a in 0.25 0.5 0.75 0.9; do tune configs/task1/weighted.yaml t1_weighted_a$a $E "$OUT" debias.alpha=$a; done
  for k in 300 500 700; do for g in 1 4; do
    tune configs/task1/spectral.yaml t1_spectral_k${k}_g$g $E "$OUT" debias.knee_rank=$k debias.sharpness=$g
  done; done
  for b in 0.5 1.0 1.5 2.0; do tune configs/task1/calibrated.yaml t1_calibrated_b$b $E "$OUT" debias.strength=$b $CAL; done
  for b in 0.5 1.0 2.0; do
    tune configs/task1/image.yaml t1_image_cal_b$b $E "$OUT" debias.strength=$b $CAL
    tune configs/task1/image.yaml t1_image_noise_b$b $E "$OUT" debias.strength=$b debias.variance_source=noise
  done
  for b in 0.5 1.0; do
    tune configs/task1/token.yaml t1_token_ls_b$b $E "$OUT" debias.strength=$b $CAL
    tune configs/task1/token.yaml t1_token_fixed_b$b $E "$OUT" debias.strength=$b debias.token_gain=fixed $CAL
  done
  # Task 2: number of prototypes M and how their similarities are combined
  tune configs/task2/single_check.yaml t2_check_m1 $E "$OUT"           # M = 1: must equal the baseline
  for M in 2 3 4 6 8; do
    tune configs/task2/max.yaml t2_max_m$M $E "$OUT" prototypes.num_prototypes=$M
    tune configs/task2/weighted.yaml t2_weighted_m$M $E "$OUT" prototypes.num_prototypes=$M
    tune configs/task2/lse.yaml t2_lse_m${M}_t0.05 $E "$OUT" prototypes.num_prototypes=$M
  done
  for T in 0.02 0.1; do tune configs/task2/lse.yaml t2_lse_m4_t$T $E "$OUT" prototypes.lse_temperature=$T; done
  tune configs/task2/chamfer.yaml t2_chamfer_m4 $E "$OUT"
  for e in 0.02 0.05 0.1; do for r in 0.1 0.5 2.0 100.0; do
    tune configs/task2/ot.yaml t2_ot_m4_e${e}_r$r $E "$OUT" prototypes.ot_epsilon=$e prototypes.ot_rho=$r
  done; done
  for t in 0.5 0.6 0.7; do tune configs/task2/agglomerative.yaml t2_agglo_tau${t}_max $E "$OUT" prototypes.ref_tau=$t; done
  tune configs/task2/max.yaml t2_max_m4_global $E "$OUT" prototypes.include_global=true
  tune configs/task2/max.yaml t2_max_m4_groupdeb $E "$OUT" prototypes.cluster_space=debiased
  for a in 0.25 0.3 0.35; do
    tune configs/task2/max.yaml t2_max_m4_alpha$a $E "$OUT" model.merge_threshold=$a
    tune configs/task2/lse.yaml t2_lse_m4_alpha$a $E "$OUT" model.merge_threshold=$a
  done
  "$PY" scripts/select_best.py --root "$OUT" --out-dir configs/selected
  ;;
r1_combined)
  require_cuda
  E=1000
  tune configs/selected/combined.yaml c_selected $E "$OUT"
  tune configs/selected/combined.yaml c_group_debiased $E "$OUT" prototypes.cluster_space=debiased
  tune configs/selected/combined.yaml c_hard_debias $E "$OUT" debias.mode=hard           # = Task 2 alone
  for a in 0.15 0.25; do tune configs/selected/combined.yaml c_alpha$a $E "$OUT" model.merge_threshold=$a; done
  "$PY" scripts/select_best.py --root "$OUT" --out-dir configs/selected
  ;;
r1_test)
  require_cuda
  test_all configs/baseline.yaml baseline "$OUT"
  test_all configs/baseline_no_debias.yaml baseline_no_debias "$OUT"
  for t in task1 task2 combined; do test_all configs/selected/$t.yaml sel_$t "$OUT"; done
  test_all configs/baseline.yaml baseline "$OUT" data.shots=5
  test_all configs/selected/combined.yaml sel_combined "$OUT" data.shots=5
  ;;
r1_corr)
  require_cuda
  C=configs/correspondence/base_readme.yaml
  CALB="debias.calibration_path=$INSID3_CALIBRATION_PATH_BASE768"
  corr $C corr_base768_original "$OUT" correspondence.debiased=false
  corr $C corr_base768_hard "$OUT" debias.mode=hard
  corr $C corr_base768_weighted_a0.5 "$OUT" debias.mode=weighted debias.alpha=0.5
  corr $C corr_base768_spectral_k20_g1 "$OUT" debias.mode=spectral debias.knee_rank=20 debias.sharpness=1
  corr $C corr_base768_calibrated_b1 "$OUT" debias.mode=calibrated $CALB
  corr $C corr_base768_image_b1 "$OUT" debias.mode=image $CALB
  corr $C corr_base768_token_ls "$OUT" debias.mode=token $CALB
  ;;
r1_report)
  report "$OUT" results/round1
  cp configs/selected/selection_report.md results/round1/
  ;;

# ───────────────────────────── round 2 ─────────────────────────────
bias)
  require_cuda
  mkdir -p results/round2
  "$PY" scripts/analyze_positional_bias.py --config configs/base.yaml --image-dir "$INSID3_CALIB_IMAGES" \
      --calibration "$INSID3_CALIBRATION_PATH" --pairs 150 --out results/round2/positional_bias \
      --variant none:debias.mode=none --variant hard_s500: \
      --variant hard_s5:model.svd_components=5 --variant hard_s20:model.svd_components=20 \
      --variant hard_s50:model.svd_components=50 --variant hard_s900:model.svd_components=900 \
      --variant spectral_k500_g4:debias.mode=spectral,debias.knee_rank=500,debias.sharpness=4 \
      --variant calibrated_b1:debias.mode=calibrated,$CAL --variant token_ls:debias.mode=token,$CAL \
      --variant rho_0.3:debias.mode=rho_threshold,$CAL --variant random_s500:debias.mode=random \
      --variant pca_s500:debias.mode=pca,$CAL --variant pca_s20:debias.mode=pca,model.svd_components=20,$CAL \
      --variant center_only:debias.mode=none,debias.center=true,$CAL \
      --variant center_hard_s500:debias.center=true,$CAL \
      --variant hard_s500_evalbasis:model.basis_eval_mode=true
  ;;
t1)
  require_cuda
  E=4000
  tune configs/baseline.yaml baseline $E "$OUT2"
  tune configs/baseline_no_debias.yaml t1_none $E "$OUT2"
  for s in 5 10 20 50; do tune configs/baseline.yaml t1_hard_s$s $E "$OUT2" model.svd_components=$s; done
  tune configs/baseline.yaml t1_rho_t0.3 $E "$OUT2" debias.mode=rho_threshold debias.rho_threshold=0.3 $CAL
  tune configs/baseline.yaml t1_random_s500 $E "$OUT2" debias.mode=random
  tune configs/baseline.yaml t1_pca_s500 $E "$OUT2" debias.mode=pca $CAL
  tune configs/baseline.yaml t1_pca_s20 $E "$OUT2" debias.mode=pca model.svd_components=20 $CAL
  tune configs/baseline_no_debias.yaml t1_center_only $E "$OUT2" debias.center=true $CAL
  tune configs/baseline.yaml t1_center_hard_s500 $E "$OUT2" debias.center=true $CAL
  tune configs/baseline.yaml t1_hard_evalbasis $E "$OUT2" model.basis_eval_mode=true
  tune configs/task1/spectral.yaml t1_spectral_k500_g4 $E "$OUT2" debias.sharpness=4      # round-1 winner
  tune configs/task1/calibrated.yaml t1_calibrated_b1.0 $E "$OUT2" $CAL
  ;;
t2)
  require_cuda
  E=4000
  tune configs/baseline.yaml baseline $E "$OUT2"
  tune configs/task2/lse.yaml t2_lse_m2 $E "$OUT2" prototypes.num_prototypes=2              # round-1 winner
  for a in 0.1 0.15; do
    tune configs/task2/lse.yaml t2_lse_m4_alpha$a $E "$OUT2" model.merge_threshold=$a
    tune configs/task2/max.yaml t2_max_m4_alpha$a $E "$OUT2" model.merge_threshold=$a
  done
  tune configs/task2/lse.yaml t2_lse_m4_scal $E "$OUT2" prototypes.score_calibration=reference
  tune configs/task2/max.yaml t2_max_m4_scal $E "$OUT2" prototypes.score_calibration=reference
  tune configs/task2/lse.yaml t2_lse_adaptive100 $E "$OUT2" prototypes.num_prototypes=5 prototypes.min_points_per_prototype=100
  # 5-shot (1,000 episodes, 5-shot runs are slower): one clustering per shot vs all shots together
  E=1000
  tune configs/baseline.yaml baseline $E "$OUT2" data.shots=5
  tune configs/task2/lse.yaml t2_lse_m2_pershot_5s $E "$OUT2" data.shots=5 prototypes.num_prototypes=2
  tune configs/task2/lse.yaml t2_lse_m2_pooled_5s $E "$OUT2" data.shots=5 prototypes.num_prototypes=2 prototypes.multishot=pooled
  tune configs/task2/lse.yaml t2_lse_adaptive_pooled_5s $E "$OUT2" data.shots=5 prototypes.multishot=pooled \
       prototypes.num_prototypes=5 prototypes.min_points_per_prototype=100
  ;;
corr)
  require_cuda
  C=configs/correspondence/large.yaml
  corr $C corr_large1024_original "$OUT2" correspondence.debiased=false
  corr $C corr_large1024_hard "$OUT2" debias.mode=hard
  corr $C corr_large1024_hard_s20 "$OUT2" model.svd_components=20
  corr $C corr_large1024_spectral_k500_g4 "$OUT2" debias.mode=spectral debias.knee_rank=500 debias.sharpness=4
  corr $C corr_large1024_calibrated_b1 "$OUT2" debias.mode=calibrated $CAL
  corr $C corr_large1024_token_ls "$OUT2" debias.mode=token $CAL
  corr $C corr_large1024_rho_0.3 "$OUT2" debias.mode=rho_threshold $CAL
  corr $C corr_large1024_center_hard "$OUT2" debias.center=true $CAL
  corr $C corr_large1024_hard_evalbasis "$OUT2" model.basis_eval_mode=true
  ;;
attribution)
  require_cuda
  test_all configs/baseline.yaml baseline "$OUT2" data.shots=5          # same-machine reference
  test_all configs/selected/task1.yaml sel_task1 "$OUT2" data.shots=5
  test_all configs/selected/task2.yaml sel_task2 "$OUT2" data.shots=5
  ;;
consistency)
  require_cuda
  REF="${INSID3_R1_ROOT:?set INSID3_R1_ROOT to the output folder of the other machine}/baseline/coco/fold0_shot1_seed0"
  "$PY" scripts/run_segmentation.py --config configs/baseline.yaml \
      --set data.fold=0 run.output_root="$OUT2" run.exp_name=consistency_baseline run.full_diagnostics=true
  mkdir -p results/round2
  "$PY" scripts/check_consistency.py --ref "$REF" --new "$OUT2/consistency_baseline/coco/fold0_shot1_seed0" \
      --out results/round2/consistency.json || true
  ;;
report)
  "$PY" scripts/select_best.py --root "$OUT2" --out-dir configs/selected_rev --min-episodes 4000
  report "$OUT2" results/round2
  cp configs/selected_rev/selection_report.md results/round2/selection_report.md
  for t in sel_task1 sel_task2; do for f in 0 1 2 3; do
    "$PY" scripts/analyze_failures.py --a "$OUT2/baseline/pascal_part/fold${f}_shot5_seed0" \
        --b "$OUT2/$t/pascal_part/fold${f}_shot5_seed0" --out results/round2/fail_${t}_5shot_pascal_f$f
  done; done
  ;;

# ───────────────────────────── later ─────────────────────────────
seeds)
  require_cuda
  for s in 1 2; do
    DATASETS=coco test_all configs/baseline.yaml baseline "$OUT2" run.seed=$s
    DATASETS=coco test_all configs/selected/task1.yaml sel_task1 "$OUT2" run.seed=$s
    DATASETS=coco test_all configs/selected/task2.yaml sel_task2 "$OUT2" run.seed=$s
  done
  ;;
crf)
  require_cuda
  "$PY" -c "import CRF" || { echo "CRF not installed: ./setup_server.sh --with-crf"; exit 1; }
  test_all configs/baseline_crf.yaml baseline_crf "$OUT2"
  for t in task1 task2 combined; do test_all configs/selected/$t.yaml sel_${t}_crf "$OUT2" model.mask_refiner=crf; done
  ;;
bench)
  require_cuda
  export DATASETS="${BENCH_DATASETS:-isic lung suim isaid}"
  "$PY" scripts/check_env.py --datasets $DATASETS
  test_all configs/baseline.yaml baseline "$OUT2"
  for t in task1 task2 combined; do test_all configs/selected/$t.yaml sel_$t "$OUT2"; done
  ;;
timing)
  require_cuda
  for t in baseline task1 task2 combined; do
    cfg=configs/selected/$t.yaml; [[ $t == baseline ]] && cfg=configs/baseline.yaml
    "$PY" scripts/run_segmentation.py --config $cfg \
        --set run.max_episodes=200 run.exp_name=timing_$t run.output_root="$OUT/timing"
  done
  ;;
*)
  sed -n '2,36p' "$0"       # print the stage list at the top of this file
  exit 1
  ;;
esac
log "run_experiments.sh STAGE=$STAGE done"
