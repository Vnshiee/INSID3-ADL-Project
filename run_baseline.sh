#!/usr/bin/env bash
# Original INSID3 baseline.
#   STAGE=tune ./run_baseline.sh    baseline on COCO tuning episodes (reference for selection)
#   STAGE=test ./run_baseline.sh    official benchmarks ($DATASETS, all folds) + SPair-71k
#   CRF=1 ...                       use CRF refinement (paper Tab. 1 setting; needs CRF package)
#   SHOTS=5 ...                     k-shot evaluation
source "$(dirname "$0")/scripts/common.sh"
require_cuda
STAGE="${STAGE:-test}"; SHOTS="${SHOTS:-1}"
CFG=configs/baseline.yaml; [[ "${CRF:-0}" == 1 ]] && CFG=configs/baseline_crf.yaml
"$PY" scripts/verify_upstream.py
seg "$CFG" "$STAGE" data.shots="$SHOTS"
if [[ "$STAGE" == test && "$SHOTS" == 1 ]]; then
  seg configs/baseline_no_debias.yaml test data.shots=1 || true   # s = 0 reference (paper Fig. 7)
  # Semantic correspondence, README setting: original vs hard-debiased features
  "$PY" scripts/run_correspondence.py --config configs/correspondence/base_readme.yaml \
        --set correspondence.debiased=false run.exp_name=corr_base768_original
  "$PY" scripts/run_correspondence.py --config configs/correspondence/base_readme.yaml
fi
log "run_baseline.sh STAGE=$STAGE done"
