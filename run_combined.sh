#!/usr/bin/env bash
# Task 1 + Task 2.
#   STAGE=tune ./run_combined.sh   combined variants on COCO tuning episodes (needs task1+task2 tune runs)
#   STAGE=test ./run_combined.sh   configs/selected/combined.yaml on all benchmarks (1-shot and 5-shot)
source "$(dirname "$0")/scripts/common.sh"
require_cuda
STAGE="${STAGE:-tune}"
require_file configs/selected/combined.yaml "Run task1 and task2 tune stages first (scripts/select_best.py)."
case "$STAGE" in
tune)
  seg configs/selected/combined.yaml tune run.exp_name=c_selected
  seg configs/selected/combined.yaml tune prototypes.cluster_space=debiased run.exp_name=c_group_debiased
  seg configs/selected/combined.yaml tune debias.mode=hard run.exp_name=c_hard_debias        # = Task 2 only
  for a in 0.15 0.25; do seg configs/selected/combined.yaml tune model.merge_threshold=$a run.exp_name=c_alpha$a; done
  "$PY" scripts/select_best.py --root "$INSID3_OUTPUT_ROOT"
  ;;
test)
  seg configs/selected/combined.yaml test
  if [[ "${FIVE_SHOT:-1}" == 1 ]]; then
    DATASETS="${DATASETS_5SHOT:-coco pascal_part}" seg configs/selected/combined.yaml test data.shots=5
    DATASETS="${DATASETS_5SHOT:-coco pascal_part}" seg configs/baseline.yaml test data.shots=5
  fi
  ;;
*) echo "STAGE must be tune|test"; exit 1;;
esac
log "run_combined.sh STAGE=$STAGE done"
