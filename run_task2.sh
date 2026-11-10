#!/usr/bin/env bash
# Task 2 — Multi-Prototype Reference Matching.
#   STAGE=tune ./run_task2.sh   prototype count x aggregation x matching sweep on COCO tuning episodes
#   STAGE=test ./run_task2.sh   selected config (configs/selected/task2.yaml) on all benchmarks
source "$(dirname "$0")/scripts/common.sh"
require_cuda
STAGE="${STAGE:-tune}"
case "$STAGE" in
tune)
  seg configs/baseline.yaml tune
  seg configs/task2/single_check.yaml tune                       # must equal the baseline (sanity)
  for M in 2 3 4 6 8; do
    seg configs/task2/max.yaml      tune prototypes.num_prototypes=$M run.exp_name=t2_max_m$M
    seg configs/task2/weighted.yaml tune prototypes.num_prototypes=$M run.exp_name=t2_weighted_m$M
    seg configs/task2/lse.yaml      tune prototypes.num_prototypes=$M run.exp_name=t2_lse_m${M}_t0.05
  done
  for T in 0.02 0.1; do seg configs/task2/lse.yaml tune prototypes.lse_temperature=$T run.exp_name=t2_lse_m4_t$T; done
  seg configs/task2/chamfer.yaml tune
  for e in 0.02 0.05 0.1; do for r in 0.1 0.5 2.0 100.0; do
    seg configs/task2/ot.yaml tune prototypes.ot_epsilon=$e prototypes.ot_rho=$r run.exp_name=t2_ot_m4_e${e}_r$r; done; done
  for t in 0.5 0.6 0.7; do seg configs/task2/agglomerative.yaml tune prototypes.ref_tau=$t run.exp_name=t2_agglo_tau${t}_max; done
  seg configs/task2/max.yaml tune prototypes.include_global=true run.exp_name=t2_max_m4_global
  seg configs/task2/max.yaml tune prototypes.cluster_space=debiased run.exp_name=t2_max_m4_groupdeb
  # merge-threshold confound: max/lse scores are larger than the single-prototype score
  for a in 0.25 0.3 0.35; do
    seg configs/task2/max.yaml tune model.merge_threshold=$a run.exp_name=t2_max_m4_alpha$a
    seg configs/task2/lse.yaml tune model.merge_threshold=$a run.exp_name=t2_lse_m4_alpha$a
  done
  "$PY" scripts/select_best.py --root "$INSID3_OUTPUT_ROOT"
  ;;
test)
  require_file configs/selected/task2.yaml "Run STAGE=tune (writes configs/selected/task2.yaml)."
  seg configs/selected/task2.yaml test
  ;;
*) echo "STAGE must be tune|test"; exit 1;;
esac
log "run_task2.sh STAGE=$STAGE done"
