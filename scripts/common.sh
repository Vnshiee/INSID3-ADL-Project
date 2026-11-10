#!/usr/bin/env bash
# Shared helpers for the top-level run_*.sh scripts. Source, do not execute.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f "$ROOT/env.sh" ]]; then source "$ROOT/env.sh"; else source "$ROOT/env.example.sh"; fi
PY="${INSID3_PYTHON:-python}"
mkdir -p "$INSID3_OUTPUT_ROOT/logs"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$INSID3_OUTPUT_ROOT/logs/run_scripts.log"; }

require_cuda() {
  "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "ERROR: CUDA is not available to $PY. Run scripts/check_env.py; see docs/TROUBLESHOOTING.md." >&2
    exit 1; }
}

require_file() { [[ -f "$1" ]] || { echo "ERROR: missing $1. $2" >&2; exit 1; }; }

nfolds() {   # official number of folds per benchmark
  case "$1" in coco|pascal_part|paco_part) echo 4;; isaid) echo 3;; lvis) echo 10;; *) echo 1;; esac
}

# seg <config> <stage: tune|test> [extra --set overrides...]
#   tune: COCO train2014 episodes (model selection); test: official benchmarks in $DATASETS, all folds.
seg() {
  local cfg="$1" stage="$2"; shift 2
  if [[ "$stage" == "tune" ]]; then
    log "seg TUNE $cfg $*"
    "$PY" scripts/run_segmentation.py --config "$cfg" --set data.split=tune data.dataset=coco "$@"
  else
    for ds in $DATASETS; do
      log "seg TEST $cfg dataset=$ds $*"
      "$PY" scripts/run_segmentation.py --config "$cfg" --folds all --set data.dataset="$ds" "$@"
    done
  fi
}

calibrate() {   # calibrate <out.pt> [--set overrides for model size / image size]
  local out="$1"; shift
  if [[ -f "$out" ]]; then log "calibration exists: $out"; return; fi
  log "calibrating -> $out"
  "$PY" scripts/calibrate_positional.py --config configs/base.yaml --image-dir "$INSID3_CALIB_IMAGES" \
       --num-images "$INSID3_CALIB_NUM" --out "$out" "$@"
  "$PY" scripts/visualize.py calibration --calibration "$out" --out "$(dirname "$out")/figs_$(basename "$out" .pt)"
}
