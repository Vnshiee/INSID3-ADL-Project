#!/usr/bin/env bash
# Complete pipeline in the intended order. Long-running (estimate: several GPU-days for the
# full benchmark list; see docs/RESOURCE_REQUIREMENTS.md). Every step is resumable: re-running
# skips finished episodes. Use DATASETS="coco pascal_part" for a reduced test benchmark.
source "$(dirname "$0")/scripts/common.sh"
require_cuda
"$PY" scripts/check_env.py --datasets coco spair
STAGE=calibrate ./run_task1.sh
STAGE=tune      ./run_baseline.sh
STAGE=tune      ./run_task1.sh
STAGE=tune      ./run_task2.sh
STAGE=tune      ./run_combined.sh
STAGE=test      ./run_baseline.sh
STAGE=test      ./run_task1.sh
STAGE=corr      ./run_task1.sh
STAGE=test      ./run_task2.sh
STAGE=test      ./run_combined.sh
./evaluate.sh
log "run_all_experiments.sh finished"
