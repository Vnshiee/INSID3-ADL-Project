#!/usr/bin/env bash
# run_test_benchmarks.sh - Run full 4-fold test benchmark evaluations (1-shot and 5-shot)
set -eo pipefail
cd "$(dirname "$0")"

if [[ -f env.sh ]]; then
  source env.sh
else
  source env.example.sh
fi

PY="${INSID3_PYTHON:-$(command -v python3 || command -v python)}"
export INSID3_PYTHON="$PY"
mkdir -p "$INSID3_OUTPUT_ROOT/logs"
LOG_FILE="$INSID3_OUTPUT_ROOT/logs/test_benchmarks_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "================================================================="
log " Starting Full 4-Fold Test Benchmarks Execution"
log " Python interpreter: $PY"
log " Output directory:   $INSID3_OUTPUT_ROOT"
log " Data directory:     $INSID3_DATA_ROOT"
log "================================================================="

# Step 1: Download Pascal-Part dataset if not present (~1.5 GB)
if [[ ! -d "$INSID3_DATA_ROOT/Pascal-Part/VOCdevkit/VOC2010/JPEGImages" ]]; then
  log "Downloading & extracting Pascal-Part dataset (~1.5 GB)..."
  ./prepare_datasets.sh --yes pascal_part 2>&1 | tee -a "$LOG_FILE" || {
    log "WARNING: Pascal-Part download encountered an issue, proceeding with COCO test folds."
  }
fi

# Detect available datasets for test stage
AVAILABLE_DATASETS="coco"
if [[ -d "$INSID3_DATA_ROOT/Pascal-Part/VOCdevkit/VOC2010/JPEGImages" ]]; then
  AVAILABLE_DATASETS="coco pascal_part"
fi

export DATASETS="$AVAILABLE_DATASETS"
log "Running TEST stage across all folds for datasets: $DATASETS"

# Step 2: Baseline (all folds)
log "--- [1/4] Running Baseline TEST (all folds) ---"
STAGE=test ./run_baseline.sh 2>&1 | tee -a "$LOG_FILE"

# Step 3: Task 1 Selected Model (Spectral k=500, gamma=4, all folds)
log "--- [2/4] Running Task 1 Selected (sel_task1) TEST (all folds) ---"
STAGE=test ./run_task1.sh 2>&1 | tee -a "$LOG_FILE"

# Step 4: Task 2 Selected Model (LSE M=2, all folds)
log "--- [3/4] Running Task 2 Selected (sel_task2) TEST (all folds) ---"
STAGE=test ./run_task2.sh 2>&1 | tee -a "$LOG_FILE"

# Step 5: Combined Selected Model (all folds)
log "--- [4/4] Running Combined Selected (sel_combined) TEST (all folds) ---"
STAGE=test ./run_combined.sh 2>&1 | tee -a "$LOG_FILE"

# Step 6: Recompile all results and update summaries
log "--- Compiling final results & reports ---"
./evaluate.sh 2>&1 | tee -a "$LOG_FILE"

log "================================================================="
log " All Test Benchmarks Finished Successfully!"
log "================================================================="
