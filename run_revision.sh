#!/usr/bin/env bash
# Revision experiments (second round), in two phases. Guide: docs/RUN_GUIDE.md.
# All runs are resumable: re-run a stage after an interruption.
#
# PHASE 1 — core questions, COCO-20i + PASCAL-Part + SPair-71k only
#   STAGE=sanity        tests, upstream check, data check, 20-episode smoke run
#   STAGE=consistency   re-run baseline COCO fold 0 and compare it episode by episode with the same run from
#                       another machine in $INSID3_R1_ROOT (-> results/consistency.json). Different GPUs change
#                       individual episodes, so every stage computes its own baseline on the machine it runs on.
#   STAGE=calibrate     positional calibration incl. feature mean/covariance (ViT-L/1024, ViT-B/768)
#   STAGE=bias          residual positional bias per debiasing variant (minutes)
#   STAGE=t1            Task-1 controls + re-check of the first-round winner (4000 COCO tuning episodes)
#   STAGE=t2            Task-2 fixes: lower alpha, score calibration, adaptive M, pooled 5-shot
#   STAGE=corr          SPair-71k at the segmentation backbone (ViT-L/1024), all key Task-1 variants
#   STAGE=attribution   5-shot test of the first-round sel_task1 / sel_task2 (who causes the 5-shot drop?)
#   STAGE=select        selection on this round's tuning runs -> configs/selected_rev/
#   STAGE=report        tables + paired bootstrap (CPU only); re-run after every stage you add
#
# PHASE 2 — later, results are appended to the same tables
#   STAGE=test_rev      configs/selected_rev/* on COCO-20i + PASCAL-Part (only if they differ from round 1)
#   STAGE=seeds         extra episode seeds 1-2 on COCO for baseline and the selected methods
#   STAGE=crf           CRF setting of paper Tab. 1 (needs ./setup_server.sh --with-crf)
#   STAGE=bench         further benchmarks: BENCH_DATASETS="isic lung suim isaid", later "lvis paco_part permis"
#   STAGE=timing        runtime/memory, 200 episodes per method, with nothing else running on the GPU
source "$(dirname "$0")/scripts/common.sh"
STAGE="${STAGE:-}"
TUNE_EPS="${TUNE_EPS:-4000}"
R1=configs/selected          # first-round (pre-registered) selections, already evaluated on the test sets
REV=configs/selected_rev
CAL="$INSID3_CALIBRATION_PATH"
CALB="$INSID3_CALIBRATION_PATH_BASE768"
C="debias.calibration_path=$CAL"
export DATASETS="${DATASETS:-coco pascal_part}"

tune() {   # tune <config> <exp_name> [overrides]   (COCO tuning split, diagnostics on)
  local cfg="$1" name="$2"; shift 2
  seg "$cfg" tune data.tune_episodes="$TUNE_EPS" run.exp_name="$name" run.full_diagnostics=true "$@"
}
test_run() {   # test_run <config> <exp_name> [overrides]   (all folds of $DATASETS)
  local cfg="$1" name="$2"; shift 2
  seg "$cfg" test run.exp_name="$name" run.full_diagnostics=true "$@"
}

case "$STAGE" in
# ───────────────────────────── PHASE 1 ─────────────────────────────
sanity)
  "$PY" -m pytest tests -q
  "$PY" scripts/verify_upstream.py
  "$PY" scripts/check_env.py --datasets coco pascal_part spair
  require_cuda
  "$PY" scripts/run_segmentation.py --config configs/baseline.yaml \
        --set run.max_episodes=20 run.exp_name=smoke run.save_predictions=5
  ;;
consistency)
  require_cuda
  REF="${INSID3_R1_ROOT:?set INSID3_R1_ROOT to the output folder of the reference run}/baseline/coco/fold0_shot1_seed0"
  require_file "$REF/episodes.jsonl" "INSID3_R1_ROOT must contain baseline/coco/fold0_shot1_seed0 from the other machine."
  DATASETS=coco "$PY" scripts/run_segmentation.py --config configs/baseline.yaml \
        --set data.fold=0 run.exp_name=consistency_baseline run.full_diagnostics=true
  mkdir -p results
  "$PY" scripts/check_consistency.py --ref "$REF" \
        --new "$INSID3_OUTPUT_ROOT/consistency_baseline/coco/fold0_shot1_seed0" --out results/consistency.json \
    || true
  log "consistency measured -> results/consistency.json (runs from different machines are never paired)"
  ;;
calibrate)
  require_cuda
  calibrate "$CAL"
  calibrate "$CALB" --set model.model_size=base model.image_size=768 model.svd_components=20
  ;;
bias)
  require_cuda
  "$PY" scripts/analyze_positional_bias.py --config configs/base.yaml --image-dir "$INSID3_CALIB_IMAGES" \
      --calibration "$CAL" --pairs 150 --out results/positional_bias \
      --variant none:debias.mode=none --variant hard_s500: \
      --variant hard_s5:model.svd_components=5 --variant hard_s20:model.svd_components=20 \
      --variant hard_s50:model.svd_components=50 --variant hard_s900:model.svd_components=900 \
      --variant spectral_k500_g4:debias.mode=spectral,debias.knee_rank=500,debias.sharpness=4 \
      --variant calibrated_b1:debias.mode=calibrated,$C --variant token_ls:debias.mode=token,$C \
      --variant rho_0.3:debias.mode=rho_threshold,$C --variant random_s500:debias.mode=random \
      --variant pca_s500:debias.mode=pca,$C --variant pca_s20:debias.mode=pca,model.svd_components=20,$C \
      --variant center_only:debias.mode=none,debias.center=true,$C \
      --variant center_hard_s500:debias.center=true,$C \
      --variant hard_s500_evalbasis:model.basis_eval_mode=true
  ;;
t1)
  require_cuda
  require_file "$CAL" "Run STAGE=calibrate first."
  tune configs/baseline.yaml baseline
  tune configs/baseline_no_debias.yaml t1_none
  for s in 5 10 20 50; do tune configs/baseline.yaml t1_hard_s$s model.svd_components=$s; done
  tune configs/baseline.yaml t1_rho_t0.3 debias.mode=rho_threshold debias.rho_threshold=0.3 $C
  tune configs/baseline.yaml t1_random_s500 debias.mode=random
  tune configs/baseline.yaml t1_pca_s500 debias.mode=pca $C
  tune configs/baseline.yaml t1_pca_s20 debias.mode=pca model.svd_components=20 $C
  tune configs/baseline_no_debias.yaml t1_center_only debias.center=true $C
  tune configs/baseline.yaml t1_center_hard_s500 debias.center=true $C
  tune configs/baseline.yaml t1_hard_evalbasis model.basis_eval_mode=true    # basis computed in eval mode
  tune configs/task1/spectral.yaml t1_spectral_k500_g4 debias.sharpness=4      # first-round winner, re-check
  tune configs/task1/calibrated.yaml t1_calibrated_b1.0 $C
  ;;
t2)
  require_cuda
  tune configs/baseline.yaml baseline
  tune configs/task2/lse.yaml t2_lse_m2 prototypes.num_prototypes=2                # first-round winner, re-check
  for a in 0.1 0.15; do
    tune configs/task2/lse.yaml t2_lse_m4_alpha$a model.merge_threshold=$a
    tune configs/task2/max.yaml t2_max_m4_alpha$a model.merge_threshold=$a
  done
  tune configs/task2/lse.yaml t2_lse_m4_scal prototypes.score_calibration=reference
  tune configs/task2/max.yaml t2_max_m4_scal prototypes.score_calibration=reference
  tune configs/task2/lse.yaml t2_lse_adaptive100 prototypes.num_prototypes=5 prototypes.min_points_per_prototype=100
  # 5-shot: per-shot vs pooled prototypes (ASGNet clusters all shots together)
  TUNE_EPS="${TUNE_EPS_5SHOT:-1000}"
  tune configs/baseline.yaml baseline data.shots=5
  tune configs/task2/lse.yaml t2_lse_m2_pershot_5s data.shots=5 prototypes.num_prototypes=2
  tune configs/task2/lse.yaml t2_lse_m2_pooled_5s data.shots=5 prototypes.num_prototypes=2 prototypes.multishot=pooled
  tune configs/task2/lse.yaml t2_lse_adaptive_pooled_5s data.shots=5 prototypes.multishot=pooled \
       prototypes.num_prototypes=5 prototypes.min_points_per_prototype=100
  ;;
corr)
  require_cuda
  SPECS=("original|correspondence.debiased=false" "hard|debias.mode=hard" "hard_s20|model.svd_components=20"
         "spectral_k500_g4|debias.mode=spectral debias.knee_rank=500 debias.sharpness=4"
         "calibrated_b1|debias.mode=calibrated $C" "token_ls|debias.mode=token $C"
         "rho_0.3|debias.mode=rho_threshold $C" "center_hard|debias.center=true $C"
         "hard_evalbasis|model.basis_eval_mode=true")
  for spec in "${SPECS[@]}"; do
    name="${spec%%|*}"; read -r -a ov <<< "${spec#*|}"
    "$PY" scripts/run_correspondence.py --config configs/correspondence/large.yaml --set "${ov[@]}" \
          run.exp_name=corr_large1024_$name
  done
  ;;
attribution)
  require_cuda
  test_run configs/baseline.yaml baseline data.shots=5          # same-machine reference
  test_run $R1/task1.yaml sel_task1 data.shots=5
  test_run $R1/task2.yaml sel_task2 data.shots=5
  ;;
select)
  "$PY" scripts/select_best.py --root "$INSID3_OUTPUT_ROOT" --out-dir $REV --min-episodes "$TUNE_EPS"
  ;;
report)
  mkdir -p results
  "$PY" scripts/aggregate_results.py --root "$INSID3_OUTPUT_ROOT" --out results/summary
  "$PY" scripts/compare_all.py --root "$INSID3_OUTPUT_ROOT" --reference baseline --out results/comparisons
  [[ -f $REV/selection_report.md ]] && cp $REV/selection_report.md results/revision_selection_report.md
  [[ -d "$INSID3_OUTPUT_ROOT/timing" ]] && \
    "$PY" scripts/aggregate_results.py --root "$INSID3_OUTPUT_ROOT/timing" --out results/timing
  for f in 0 1 2 3; do
    for e in sel_task2 sel_combined; do
      a="$INSID3_OUTPUT_ROOT/baseline/pascal_part/fold${f}_shot5_seed0"; b="$INSID3_OUTPUT_ROOT/$e/pascal_part/fold${f}_shot5_seed0"
      if [[ -d "$a" && -d "$b" ]]; then "$PY" scripts/analyze_failures.py --a "$a" --b "$b" --out results/fail_${e}_5shot_pascal_f$f; fi
    done
  done
  log "report written to results/"
  ;;
# ───────────────────────────── PHASE 2 ─────────────────────────────
test_rev)
  require_cuda
  for t in task1 task2 combined; do require_file $REV/$t.yaml "Run STAGE=select first."; done
  test_run configs/baseline.yaml baseline                       # same-machine references
  test_run configs/baseline.yaml baseline data.shots=5
  for t in task1 task2 combined; do
    test_run $REV/$t.yaml rev_$t
    test_run $REV/$t.yaml rev_$t data.shots=5
  done
  ;;
seeds)
  require_cuda
  for s in 1 2; do
    for pair in "configs/baseline.yaml baseline" "$R1/task1.yaml sel_task1" "$R1/task2.yaml sel_task2"; do
      set -- $pair; DATASETS=coco test_run "$1" "$2" run.seed=$s
    done
  done
  ;;
crf)
  require_cuda
  "$PY" -c "import CRF" || { echo "CRF not installed: ./setup_server.sh --with-crf"; exit 1; }
  test_run configs/baseline_crf.yaml baseline_crf
  for t in task1 task2 combined; do test_run $R1/$t.yaml sel_${t}_crf model.mask_refiner=crf; done
  ;;
bench)
  require_cuda
  export DATASETS="${BENCH_DATASETS:-isic lung suim isaid}"
  "$PY" scripts/check_env.py --datasets $DATASETS
  test_run configs/baseline.yaml baseline
  for t in task1 task2 combined; do test_run $R1/$t.yaml sel_$t; done
  ;;
timing)
  require_cuda
  for pair in "configs/baseline.yaml baseline" "$R1/task1.yaml task1" "$R1/task2.yaml task2" "$R1/combined.yaml combined"; do
    set -- $pair
    "$PY" scripts/run_segmentation.py --config "$1" --set run.max_episodes=200 \
          run.exp_name=timing_$2 run.output_root="$INSID3_OUTPUT_ROOT/timing"
  done
  ;;
*) sed -n '2,23p' "$0"; exit 1;;
esac
log "run_revision.sh STAGE=$STAGE done"
