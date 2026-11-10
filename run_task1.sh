#!/usr/bin/env bash
# Task 1 — Adaptive Positional Debiasing.
#   STAGE=calibrate ./run_task1.sh   measure positional components (needed by calibrated/image/token)
#   STAGE=tune      ./run_task1.sh   ablation sweep on COCO tuning episodes (model selection)
#   STAGE=test      ./run_task1.sh   selected config (configs/selected/task1.yaml) on all benchmarks
#   STAGE=corr      ./run_task1.sh   SPair-71k correspondence for every Task-1 mode
# Run order: calibrate -> tune -> (scripts/select_best.py) -> test, corr.
source "$(dirname "$0")/scripts/common.sh"
require_cuda
STAGE="${STAGE:-tune}"
case "$STAGE" in
calibrate)
  calibrate "$INSID3_CALIBRATION_PATH"
  calibrate "$INSID3_CALIBRATION_PATH_BASE768" --set model.model_size=base model.image_size=768 model.svd_components=20
  ;;
tune)
  require_file "$INSID3_CALIBRATION_PATH" "Run STAGE=calibrate first."
  seg configs/baseline.yaml tune
  seg configs/baseline_no_debias.yaml tune run.exp_name=t1_none
  for s in 100 300 700 900; do seg configs/baseline.yaml tune model.svd_components=$s run.exp_name=t1_hard_s$s; done
  for a in 0.25 0.5 0.75 0.9; do seg configs/task1/weighted.yaml tune debias.alpha=$a run.exp_name=t1_weighted_a$a; done
  for k in 300 500 700; do for g in 1 4; do
    seg configs/task1/spectral.yaml tune debias.knee_rank=$k debias.sharpness=$g run.exp_name=t1_spectral_k${k}_g$g; done; done
  for b in 0.5 1.0 1.5 2.0; do seg configs/task1/calibrated.yaml tune debias.strength=$b run.exp_name=t1_calibrated_b$b; done
  for b in 0.5 1.0 2.0; do
    seg configs/task1/image.yaml tune debias.strength=$b run.exp_name=t1_image_cal_b$b
    seg configs/task1/image.yaml tune debias.strength=$b debias.variance_source=noise run.exp_name=t1_image_noise_b$b
  done
  for b in 0.5 1.0; do
    seg configs/task1/token.yaml tune debias.strength=$b run.exp_name=t1_token_ls_b$b
    seg configs/task1/token.yaml tune debias.strength=$b debias.token_gain=fixed run.exp_name=t1_token_fixed_b$b
  done
  "$PY" scripts/select_best.py --root "$INSID3_OUTPUT_ROOT"
  ;;
test)
  require_file configs/selected/task1.yaml "Run STAGE=tune (writes configs/selected/task1.yaml)."
  seg configs/selected/task1.yaml test
  ;;
corr)
  require_file "$INSID3_CALIBRATION_PATH_BASE768" "Run STAGE=calibrate first."
  C=configs/correspondence/base_readme.yaml
  # "<exp_name suffix>|<overrides>"  (README setting: ViT-B, 768 px, s = 20)
  SPECS=("hard|debias.mode=hard"
         "weighted_a0.5|debias.mode=weighted debias.alpha=0.5"
         "spectral_k20_g1|debias.mode=spectral debias.knee_rank=20 debias.sharpness=1"
         "calibrated_b1|debias.mode=calibrated"
         "image_b1|debias.mode=image"
         "token_ls|debias.mode=token")
  for spec in "${SPECS[@]}"; do
    name="${spec%%|*}"; read -r -a ov <<< "${spec#*|}"
    "$PY" scripts/run_correspondence.py --config $C --set "${ov[@]}" run.exp_name=corr_base768_$name
  done
  ;;
*) echo "STAGE must be calibrate|tune|test|corr"; exit 1;;
esac
log "run_task1.sh STAGE=$STAGE done"
