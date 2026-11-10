#!/usr/bin/env bash
# run_pipeline.sh - Orchestrate end-to-end dataset prep, tuning sweeps, correspondence, and reporting
set -eo pipefail
cd "$(dirname "$0")"

# Source environment configuration
if [[ -f env.sh ]]; then
  source env.sh
else
  source env.example.sh
fi

PY="${INSID3_PYTHON:-$(command -v python3 || command -v python)}"
export INSID3_PYTHON="$PY"
mkdir -p "$INSID3_OUTPUT_ROOT/logs"
LOG_FILE="$INSID3_OUTPUT_ROOT/logs/pipeline_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "================================================================="
log " Starting INSID3 Research Execution Pipeline"
log " Python interpreter: $PY"
log " Output directory:   $INSID3_OUTPUT_ROOT"
log " Data directory:     $INSID3_DATA_ROOT"
log "================================================================="

# Step 0: Initial environment & GPU sanity check
log "--- [Step 0/5] Verifying environment & upstream integrity ---"
"$PY" scripts/verify_upstream.py | tee -a "$LOG_FILE"
"$PY" scripts/check_env.py | tee -a "$LOG_FILE"

# Step 1: Ensure required datasets are present
log "--- [Step 1/5] Preparing datasets (spair + coco) ---"
if [[ ! -d "$INSID3_DATA_ROOT/SPair-71k/JPEGImages" ]]; then
  log "Downloading SPair-71k..."
  ./prepare_datasets.sh --yes spair 2>&1 | tee -a "$LOG_FILE"
else
  log "SPair-71k dataset already present."
fi

if [[ ! -d "$INSID3_DATA_ROOT/COCO2014/train2014" ]]; then
  log "Downloading COCO 2014 (this may take a while depending on network speed)..."
  ./prepare_datasets.sh --yes coco 2>&1 | tee -a "$LOG_FILE"
else
  log "COCO 2014 dataset already present."
fi

# Step 2: Task 1 Calibration
log "--- [Step 2/5] Running Task 1 Positional Calibration ---"
STAGE=calibrate ./run_task1.sh 2>&1 | tee -a "$LOG_FILE"

# Step 3: Tuning Sweeps (Baseline, Task 1, Task 2, Combined)
log "--- [Step 3/5] Running Tuning Sweeps on COCO tuning split ---"
log ">> Running Baseline tune..."
STAGE=tune ./run_baseline.sh 2>&1 | tee -a "$LOG_FILE"

log ">> Running Task 1 tune sweep..."
STAGE=tune ./run_task1.sh 2>&1 | tee -a "$LOG_FILE"

log ">> Running Task 2 tune sweep..."
STAGE=tune ./run_task2.sh 2>&1 | tee -a "$LOG_FILE"

log ">> Running Combined tune sweep..."
STAGE=tune ./run_combined.sh 2>&1 | tee -a "$LOG_FILE"

# Step 4: Semantic Correspondence on SPair-71k
log "--- [Step 4/5] Running Semantic Correspondence on SPair-71k ---"
STAGE=corr ./run_task1.sh 2>&1 | tee -a "$LOG_FILE"

# Step 5: Results Evaluation and Markdown Generation
log "--- [Step 5/5] Compiling results & reports ---"
./evaluate.sh 2>&1 | tee -a "$LOG_FILE"

log "================================================================="
log " INSID3 Research Pipeline execution completed successfully!"
log "================================================================="
