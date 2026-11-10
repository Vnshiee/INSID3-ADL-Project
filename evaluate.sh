#!/usr/bin/env bash
# Aggregate all finished runs into tables and paired-bootstrap comparisons (no GPU needed).
#   ./evaluate.sh            -> results/summary.md, results/comparisons.md, results/selection_report.md
source "$(dirname "$0")/scripts/common.sh"
mkdir -p results
"$PY" scripts/aggregate_results.py --root "$INSID3_OUTPUT_ROOT" --out results/summary
"$PY" scripts/compare_all.py --root "$INSID3_OUTPUT_ROOT" --reference baseline --out results/comparisons
[[ -f configs/selected/selection_report.md ]] && cp configs/selected/selection_report.md results/ || true
log "evaluate.sh done -> results/"
